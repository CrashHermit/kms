"""Trace capture: image redaction and the enable/flush hooks.

MLflow owns storage, and stage identity comes free from naming each stage's dspy.Module
subclass after its stage (pinned in `test_datasets.py`). What is left here is the one thing
MLflow has no built-in answer for — keeping page images out of the store — plus the guards
that keep capture from ever breaking a run. Nothing here needs mlflow installed: the span
processor is driven directly with stand-in objects.
"""

from kms.core import tracing

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
