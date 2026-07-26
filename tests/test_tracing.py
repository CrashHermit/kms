"""Trace capture: value serialisation, stage naming, and the record-pairing recorder.

Everything here runs without dspy callbacks firing — the recorder's two hooks are called
directly with stand-in objects, so the pairing logic is tested without an LLM.
"""

import json

from kms.core import tracing

# --- value serialisation ---


class _Model:
    """Stands in for a pydantic boundary DTO (the window/member models)."""

    def model_dump(self):
        return {'position': 0, 'content': 'x'}


class Image:
    """Stands in for dspy.Image, which `_plain` matches by class name so that `core`
    need not import a dspy symbol the test stub may not define."""


def test_plain_passes_through_json_natives():
    assert tracing._plain('a') == 'a'
    assert tracing._plain(3) == 3
    assert tracing._plain(True) is True
    assert tracing._plain(None) is None


def test_plain_unwraps_a_pydantic_model():
    assert tracing._plain(_Model()) == {'position': 0, 'content': 'x'}


def test_plain_replaces_an_image_with_a_placeholder():
    # the bytes are large and reconstructable from the PDF; the text signal is what trains
    assert tracing._plain(Image()) == '<image>'


def test_plain_recurses_into_lists_and_dicts():
    assert tracing._plain([_Model()]) == [{'position': 0, 'content': 'x'}]
    assert tracing._plain({'k': [1, 'a']}) == {'k': [1, 'a']}


def test_plain_falls_back_to_str_for_anything_else():
    assert tracing._plain(object()).startswith('<object')


# --- stage naming ---


class _Signature:
    """A signature defined in a kms module, as a plain Predict would carry it."""

    __module__ = 'kms.entity.group_finder'
    instructions = 'Find the BOUNDARIES.'
    input_fields = {}


def test_stage_comes_from_the_signatures_defining_module():
    assert tracing._stage_of(_Signature) == 'group_finder'


def test_stage_falls_back_to_unknown_for_an_unmatched_signature():
    class Foreign:
        __module__ = 'dspy.signatures.signature'
        instructions = 'not one of ours'
        input_fields = {}

    assert tracing._stage_of(Foreign) == 'unknown'


def test_stage_of_a_real_signature_resolves_by_instructions():
    # ChainOfThought rebuilds the signature to add `reasoning`, losing __module__ but
    # copying the instructions verbatim — that is what the registry matches on.
    from kms.entity import role_typer

    class Rebuilt:
        __module__ = 'dspy.signatures.signature'
        instructions = role_typer.Classify.instructions
        input_fields = {}

    assert tracing._stage_of(Rebuilt) == 'role_typer'


# --- the recorder ---


class _Predict:
    """A predictor: exposes `.signature`, so it is the one that gets recorded."""

    signature = _Signature


class _Wrapper:
    """A ChainOfThought-like wrapper: no `.signature`, so it must be skipped."""


class _Prediction(dict):
    """Stands in for dspy.Prediction, which exposes its fields via keys()."""


def _recorder(tmp_path):
    return tracing._Recorder(tmp_path)


def test_records_one_line_pairing_inputs_with_outputs(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.on_module_start(
        'c1', _Predict(), {'args': (), 'kwargs': {'contents': 'proof text'}}
    )
    recorder.on_module_end('c1', _Prediction(role='procedure'))

    lines = (tmp_path / 'group_finder.jsonl').read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record['stage'] == 'group_finder'
    assert record['inputs'] == {'contents': 'proof text'}
    assert record['outputs'] == {'role': 'procedure'}


def test_a_wrapper_without_a_signature_is_not_recorded(tmp_path):
    # ChainOfThought fires a callback AND wraps an inner Predict; recording both would
    # double every line.
    recorder = _recorder(tmp_path)
    recorder.on_module_start('c1', _Wrapper(), {'kwargs': {'a': 1}})
    recorder.on_module_end('c1', _Prediction(role='entity'))
    assert not list(tmp_path.glob('*.jsonl'))


def test_a_failed_call_is_not_recorded(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.on_module_start('c1', _Predict(), {'kwargs': {'a': 1}})
    recorder.on_module_end('c1', None, ValueError('boom'))
    assert not list(tmp_path.glob('*.jsonl'))


def test_an_unpaired_end_is_ignored(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.on_module_end('never-started', _Prediction(role='entity'))
    assert not list(tmp_path.glob('*.jsonl'))


def test_each_stage_gets_its_own_file_and_appends(tmp_path):
    recorder = _recorder(tmp_path)
    for call_id in ('c1', 'c2'):
        recorder.on_module_start(call_id, _Predict(), {'kwargs': {'a': 1}})
        recorder.on_module_end(call_id, _Prediction(x='y'))
    assert len((tmp_path / 'group_finder.jsonl').read_text().splitlines()) == 2
    assert recorder.written == 2


def test_enable_from_env_is_a_noop_when_unset(monkeypatch):
    monkeypatch.delenv(tracing.TRACE_DIR_ENV, raising=False)
    tracing._recorder = None
    tracing.enable_from_env()
    assert tracing._recorder is None
    assert tracing.written() == 0
