"""Trace capture: stage naming, image redaction, and the enable/flush hooks.

MLflow owns storage now, so there is no recorder to exercise — what is left is the two
things MLflow does not solve (which stage a span belongs to, and keeping page images out
of the store) plus the guards that keep capture from ever breaking a run. Nothing here
needs mlflow installed: the tagger and the span processor are driven directly with
stand-in objects.
"""

from kms.core import tracing

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
    # copying the instructions verbatim — that is what the registry matches on. Ten of the
    # twelve stages go through ChainOfThought, so this path carries most of the pipeline.
    from kms.entity import role_typer

    class Rebuilt:
        __module__ = 'dspy.signatures.signature'
        instructions = role_typer.Classify.instructions
        input_fields = {}

    assert tracing._stage_of(Rebuilt) == 'role_typer'


# --- image redaction ---


class Image:
    """Stands in for dspy.Image, which `_is_image` matches by class name so that `core`
    need not import a dspy symbol the test stub may not define."""


class _Span:
    """Stands in for an MLflow span: inputs in, `set_inputs` to rewrite them."""

    def __init__(self, inputs):
        self.inputs = inputs

    def set_inputs(self, value):
        self.inputs = value


def test_is_image_matches_a_dspy_image_and_its_serialised_forms():
    assert tracing._is_image(Image())
    assert tracing._is_image(
        '<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "image_url", "image_url": {}}]'
    )
    assert tracing._is_image('data:image/png;base64,iVBORw0KGgo=')


def test_is_image_does_not_match_prose_that_merely_mentions_base64():
    # a page of a textbook *about* base64 is a training example, not an image
    assert not tracing._is_image(
        'A base64, encoding maps three octets to four characters.'
    )
    assert not tracing._is_image('The image_url field is optional.')


def test_strip_images_replaces_only_the_image_input():
    span = _Span({'page_image': Image(), 'transcription': 'Theorem 1.l'})
    tracing._strip_images(span)
    assert span.inputs == {
        'page_image': '<image>',
        'transcription': 'Theorem 1.l',
    }


def test_strip_images_leaves_an_image_free_span_untouched():
    original = {'current_nodes': [{'position': 0, 'content': 'x'}]}
    span = _Span(dict(original))
    tracing._strip_images(span)
    assert span.inputs == original


def test_strip_images_never_raises_on_a_hostile_span():
    class Hostile:
        @property
        def inputs(self):
            raise RuntimeError('boom')

    tracing._strip_images(Hostile())  # must not propagate


# --- the tagger ---


class _Predict:
    """A predictor: exposes `.signature`, so it is the one whose span gets tagged."""

    signature = _Signature


class _Wrapper:
    """A ChainOfThought-like wrapper: no `.signature`, so it must be skipped."""


def test_the_tagger_skips_a_wrapper_without_a_signature(monkeypatch):
    # ChainOfThought fires a callback AND wraps an inner Predict; tagging both would put
    # the stage on two spans for one logical call.
    calls = []
    monkeypatch.setattr(
        tracing, '_stage_of', lambda sig: calls.append(sig) or 'x'
    )
    tracing._StageTagger().on_module_start('c1', _Wrapper(), {})
    assert calls == []


def test_the_tagger_never_raises_when_mlflow_is_absent():
    # capture is a convenience; it must never break a run
    tracing._StageTagger().on_module_start('c1', _Predict(), {'kwargs': {}})


# --- enable / flush ---


def test_enable_from_env_is_a_noop_when_unset(monkeypatch):
    monkeypatch.delenv(tracing.TRACE_DIR_ENV, raising=False)
    monkeypatch.setattr(tracing, '_enabled', False)
    tracing.enable_from_env()
    assert tracing._enabled is False


def test_enable_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing, '_enabled', True)
    tracing.enable(tmp_path)  # already on: must not re-register hooks
    assert tracing._enabled is True


def test_flush_is_a_noop_when_capture_is_off(monkeypatch):
    monkeypatch.setattr(tracing, '_enabled', False)
    tracing.flush()  # must not import mlflow or raise


def test_store_uri_is_a_sqlite_file_inside_the_directory(tmp_path):
    uri = tracing.store_uri(tmp_path)
    assert uri.startswith('sqlite:///')
    assert uri.endswith('mlruns.db')
