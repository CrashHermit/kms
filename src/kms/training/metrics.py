r"""Stage-specific metric functions for DSPy prompt optimisation.

Each metric is a callable with the signature
``metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool | float``
— the contract DSPy's ``BootstrapFewShot`` and ``MIPROv2`` optimisers expect.

Every metric here follows the same pattern: a judge ``dspy.Signature`` fed through
``dspy.ChainOfThought`` with a dedicated judge LM (``metric_lm`` for text stages,
``corrector_judge_lm`` for the vision stage). The judge sees:

* the inputs the stage received,
* the expected output (from the trace — what the stage actually produced during the
  captured run), and
* the predicted output (what the optimiser's compiled candidate produced).

It returns a float score 0.0–1.0 and brief feedback. A composite ``round_trip_metric``
re-runs the whole pipeline through the optimised modules and scores the final output.

During optimisation (``trace is not None``), scores are binarised at a threshold so the
optimiser gets clean pass/fail signals for demo selection. At eval time (``trace is None``),
continuous scores are returned for inspection.
"""

from __future__ import annotations

from collections.abc import Callable

import dspy

from kms.core import llm

_THRESHOLD = 0.66
"""Scores >= this become True during optimisation; below becomes False."""


# ---------------------------------------------------------------------------
# Judge signatures
# ---------------------------------------------------------------------------


class _TextJudge(dspy.Signature):
    """Rate the quality of a predicted output against a reference. Be strict:
    small differences that change meaning are errors; cosmetic differences
    (whitespace, delimiter style) are not.

    Return a score from 0.0 (completely wrong) to 1.0 (perfect) and a short
    explanation."""

    inputs: str = dspy.InputField(
        desc='What the system was asked to process'
    )
    expected: str = dspy.InputField(
        desc='The correct / reference output'
    )
    predicted: str = dspy.InputField(
        desc='What the system actually produced'
    )
    score: float = dspy.OutputField(desc='Quality score 0.0–1.0')
    feedback: str = dspy.OutputField(
        desc='One-sentence reason for the score'
    )


class _CorrectorJudge(dspy.Signature):
    """You are a meticulous proofreading judge. You see a page image, its original
    OCR transcription, a reference corrected version, and a candidate correction.
    Score how faithful the candidate is to the reference — did it fix the same
    genuine errors without making new ones?

    Return a score from 0.0 (completely wrong) to 1.0 (perfect) and a short
    explanation."""

    page_image: dspy.Image = dspy.InputField(
        desc='The textbook page image (ground truth)'
    )
    transcription: str = dspy.InputField(
        desc='The original OCR transcription'
    )
    expected: str = dspy.InputField(
        desc='The reference corrected transcription'
    )
    predicted: str = dspy.InputField(
        desc='The candidate corrected transcription to judge'
    )
    score: float = dspy.OutputField(desc='Faithfulness score 0.0–1.0')
    feedback: str = dspy.OutputField(
        desc='One-sentence reason for the score'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _judge(
    example: dspy.Example,
    prediction: dspy.Prediction,
    input_keys: list[str],
    expected_keys: list[str],
    predicted_key: str,
    judge_signature: type[dspy.Signature],
    judge_lm: Callable[[], dspy.LM],
    trace: object | None,
    _threshold: float = _THRESHOLD,
    **extra_kwargs,
) -> float | bool:
    """Run a judge over one example/prediction pair.

    Args:
        example: The gold example from the traces (inputs + expected outputs).
        prediction: What the optimiser's candidate produced.
        input_keys: The input field names to pass as ``inputs`` to the judge.
        expected_keys: The output field names to read from ``example`` as the
            expected / reference.
        predicted_key: The output field name to read from ``prediction``.
        judge_signature: The dspy.Signature subclass that defines the judge.
        judge_lm: Factory for the judge's LM (e.g. ``llm.metric_lm``).
        trace: Passed through from the metric; when not None we binarise.
        _threshold: The score above which we return True when tracing.
        **extra_kwargs: Extra keyword fields to pass to the judge (e.g.
            ``page_image=...`` for the corrector VLM judge).
    """
    serialised = _serialise(example, input_keys, expected_keys)
    predicted_text = _serialise_attr(prediction, predicted_key)
    if not predicted_text:
        return True if trace is not None else 1.0  # empty → no penalty

    with dspy.context(lm=judge_lm()):
        judge = dspy.ChainOfThought(judge_signature)
        result = judge(
            inputs=serialised['inputs'],
            expected=serialised['expected'],
            predicted=predicted_text,
            **extra_kwargs,
        )
    score = float(getattr(result, 'score', 0.0) or 0.0)
    if trace is not None:
        return score >= _threshold
    return score


def _serialise(
    example: dspy.Example,
    input_keys: list[str],
    expected_keys: list[str],
) -> dict[str, str]:
    """Pull the input and expected fields from an example as display strings."""
    return {
        'inputs': _serialise_attrs(example, input_keys),
        'expected': _serialise_attrs(example, expected_keys),
    }


def _serialise_attrs(obj: object, keys: list[str]) -> str:
    """Format named attributes of *obj* for human/judge readability."""
    parts: list[str] = []
    for key in keys:
        value = _serialise_attr(obj, key)
        if value:
            parts.append(f'{key}: {value}')
    return '\n\n'.join(parts)


def _serialise_attr(obj: object, key: str) -> str:
    """One attribute, stringified. Complex types get a compact ``repr``."""
    value = getattr(obj, key, None)
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    # Lists, dicts, pydantic models — just dump.
    return _compact_repr(value)


def _compact_repr(value: object) -> str:
    """``repr`` shortened to at most 3000 characters."""
    text = repr(value)
    if len(text) > 3000:
        text = text[:2997] + '...'
    return text


# ---------------------------------------------------------------------------
# Stage metrics
# ---------------------------------------------------------------------------


def corrector_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge the corrector's output against the page image and reference.

    This is a VLM metric — it uses ``corrector_judge_lm`` (MiMo-V2.5) so the
    judge can compare the candidate correction against the actual page image.
    Requires ``KMS_TRACE_STRIP_IMAGES=0`` during trace collection, otherwise
    the page image is ``'<image>'`` and the judge cannot see it.
    """
    page_image = _coerce_image(example, 'page_image')
    return _judge(
        example,
        prediction,
        input_keys=['transcription'],
        expected_keys=['corrected'],
        predicted_key='corrected',
        judge_signature=_CorrectorJudge,
        judge_lm=llm.corrector_judge_lm,
        trace=trace,
        page_image=page_image,
    )


def _coerce_image(
    example: dspy.Example, key: str
) -> dspy.Image:
    """Turn a traced image value back into a ``dspy.Image``.

    Traces store images as base64 data-URL strings (when stripping is off);
    the VLM judge needs a ``dspy.Image``. If the value is already a
    ``dspy.Image`` or the placeholder ``'<image>'``, return it as-is — the
    VLM will see the placeholder text instead of the real image.
    """
    value = getattr(example, key, None)
    if isinstance(value, dspy.Image):
        return value
    if isinstance(value, str) and value.startswith('data:image/'):
        return dspy.Image(url=value)
    # Placeholder or missing — return an empty image so the judge still runs
    # but will likely score low (no image to check against).
    return dspy.Image(url='')


def extractor_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the extractor's structural parse matches the reference."""
    return _judge(
        example,
        prediction,
        input_keys=['segment_markdown'],
        expected_keys=['nodes'],
        predicted_key='nodes',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


def seam_merger_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the seam merger's merge decision was correct."""
    return _judge(
        example,
        prediction,
        input_keys=[
            'top_node_context',
            'top_bottom_edge_node',
            'bottom_top_edge_node',
            'bottom_node_context',
        ],
        expected_keys=['node'],
        predicted_key='node',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


def splitter_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the splitter identified the right packed nodes to split."""
    return _judge(
        example,
        prediction,
        input_keys=['current_nodes'],
        expected_keys=['splits'],
        predicted_key='splits',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


def instruction_finder_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the instruction finder tagged the correct lead-in positions."""
    return _judge(
        example,
        prediction,
        input_keys=['current_nodes'],
        expected_keys=['instruction_positions'],
        predicted_key='instruction_positions',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


def instruction_distributor_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the distributor assigned the right governance range."""
    return _judge(
        example,
        prediction,
        input_keys=['lead_in', 'following_problems'],
        expected_keys=['instruction', 'governed_positions'],
        predicted_key='governed_positions',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


def pedagogical_component_finder_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the span boundaries are correct."""
    return _judge(
        example,
        prediction,
        input_keys=['current_nodes'],
        expected_keys=['spans'],
        predicted_key='spans',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


def role_typer_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the span was classified as the correct role."""
    return _judge(
        example,
        prediction,
        input_keys=['contents'],
        expected_keys=['role'],
        predicted_key='role',
        judge_signature=_TextJudge,
        judge_lm=llm.metric_lm,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------

# Stage name (as in datasets.examples_by_stage) -> metric function.
STAGE_METRICS: dict[str, Callable] = {
    'corrector': corrector_metric,
    'extractor': extractor_metric,
    'seam_merger': seam_merger_metric,
    'splitter': splitter_metric,
    'instruction_finder': instruction_finder_metric,
    'instruction_distributor': instruction_distributor_metric,
    'pedagogical_component_finder': pedagogical_component_finder_metric,
    'role_typer': role_typer_metric,
}
"""Maps a stage's trace name to its metric, for programmatic lookup."""
