"""Reading traces back as `dspy.Example`s, grouped by stage.

The MLflow store is stood in for: span selection and the example-building loop are the logic
worth pinning, and both take plain objects. Loading from a real store is exercised end to end
elsewhere (it needs the mlflow extra); these run anywhere.
"""

import importlib
import pkgutil

import pytest

from kms.core import datasets


class _Span:
    """Stands in for an MLflow span."""

    def __init__(self, name, parent_id='p', inputs=None, outputs=None):
        self.name = name
        self.parent_id = parent_id
        self.inputs = inputs
        self.outputs = outputs


class _Trace:
    """Stands in for an MLflow trace: a bag of spans under `.data.spans`."""

    def __init__(self, *spans):
        self.data = type('Data', (), {'spans': list(spans)})()


def _call(stage_class, inputs, outputs):
    """The span shape one stage call produces: a named root over a Predict over adapters."""
    return _Trace(
        _Span(f'{stage_class}.forward', parent_id=None, inputs=inputs),
        _Span('ChainOfThought.forward', inputs=inputs),
        _Span('Predict.forward', inputs=inputs, outputs=outputs),
        _Span('ChatAdapter.format', inputs={'signature': 'x -> y'}),
        _Span('LM.__call__', inputs={'messages': []}, outputs=['...']),
    )


# --- the naming contract ---


def test_stage_name_snake_cases_a_module_class():
    assert datasets.stage_name('GroupFinder') == 'group_finder'
    assert datasets.stage_name('Corrector') == 'corrector'
    assert datasets.stage_name('InstructionDistributor') == (
        'instruction_distributor'
    )


def _stage_modules():
    """Every kms stage module that defines a dspy.Module subclass."""
    import dspy

    import kms

    found = []
    for info in pkgutil.walk_packages(kms.__path__, prefix='kms.'):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, dspy.Module)
                and value is not dspy.Module
                and value.__module__ == info.name
            ):
                found.append((info.name, value))
    return found


def test_every_stage_class_is_named_for_its_module():
    # The whole stage-identity mechanism is this convention: MLflow names the root span
    # after the class, and `stage_name` maps it back. A stage class named anything else
    # lands its examples under a key nothing looks for, silently — hence this guard.
    stages = _stage_modules()
    if not stages:
        pytest.skip('dspy is stubbed; stage classes are not importable')
    assert len(stages) >= 11
    for module_name, klass in stages:
        assert (
            datasets.stage_name(klass.__name__)
            == module_name.rsplit('.', 1)[-1]
        ), f'{klass.__name__} in {module_name} breaks the naming contract'


def test_every_stage_entry_point_is_aforward():
    # The other half of the contract. MLflow's DSPy integration is a DSPy callback, and
    # DSPy only fires callbacks for Module.__call__/acall — so a stage whose real entry
    # point is a custom method called directly produces no span named after its class,
    # and its calls land under whatever happened to be the trace root. Five stages were
    # written that way (`role`, `steps`, `identity`, `govern`, `block_type`).
    stages = _stage_modules()
    if not stages:
        pytest.skip('dspy is stubbed; stage classes are not importable')
    for module_name, klass in stages:
        own = vars(klass)
        assert 'aforward' in own or 'forward' in own, (
            f'{klass.__name__} in {module_name} exposes no aforward/forward; '
            'callers must reach it through acall() or it will not be traced'
        )


# --- span selection ---


def test_the_stage_comes_from_the_root_span_and_the_example_from_predict():
    calls = datasets._calls(
        [_call('GroupFinder', {'current_nodes': []}, {'spans': []})]
    )
    assert len(calls) == 1
    stage, span = calls[0]
    assert stage == 'group_finder'
    assert span.name == 'Predict.forward'  # not the root, not the adapters


def test_a_trace_without_a_root_is_skipped():
    orphan = _Trace(_Span('Predict.forward', inputs={'a': 1}, outputs={'b': 2}))
    assert datasets._calls([orphan]) == []


def test_a_trace_with_no_predict_span_is_skipped():
    # a stage that short-circuits before calling its predictor has nothing to learn from
    trace = _Trace(_Span('GroupFinder.forward', parent_id=None, inputs={}))
    assert datasets._calls([trace]) == []


# --- example building ---


def test_examples_are_grouped_by_stage(monkeypatch):
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _call('GroupFinder', {'current_nodes': [1]}, {'spans': []}),
            _call('GroupFinder', {'current_nodes': [2]}, {'spans': []}),
            _call('RoleTyper', {'contents': 'c'}, {'role': 'entity'}),
        ],
    )
    by_stage = datasets.examples_by_stage('traces/book')
    assert sorted(by_stage) == ['group_finder', 'role_typer']
    assert len(by_stage['group_finder']) == 2


def test_an_example_marks_the_calls_inputs_as_input_keys(monkeypatch):
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _call(
                'RoleTyper',
                {'contents': 'Theorem 1.1'},
                {'reasoning': 'it asserts', 'role': 'entity'},
            )
        ],
    )
    example = datasets.examples_by_stage('traces/book')['role_typer'][0]
    assert example.inputs().toDict() == {'contents': 'Theorem 1.1'}
    # ChainOfThought's reasoning is signal worth training on, so it stays a label
    assert example.labels().toDict() == {
        'reasoning': 'it asserts',
        'role': 'entity',
    }


def test_a_half_recorded_call_is_dropped(monkeypatch):
    # a failed call has inputs but no outputs; it trains nothing
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _call('RoleTyper', {'contents': 'c'}, None),
            _call('RoleTyper', None, {'role': 'entity'}),
        ],
    )
    assert datasets.examples_by_stage('traces/book') == {}


def test_stage_counts_summarises_a_sweep(monkeypatch):
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _call('RoleTyper', {'contents': 'a'}, {'role': 'entity'}),
            _call('RoleTyper', {'contents': 'b'}, {'role': 'procedure'}),
            _call('BlockTyper', {'contents': 'c'}, {'type': 'theorem'}),
        ],
    )
    assert datasets.stage_counts('traces/book') == {
        'block_typer': 1,
        'role_typer': 2,
    }
