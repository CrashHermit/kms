"""Shared helpers for DSPy judge metrics: scoring, serialisation, and the
common ``_judge`` dispatch that every stage metric calls."""

from collections.abc import Callable

import dspy

_PENALTY_CAP = 5.0
"""Maximum penalty beyond which the score floors at 0.0."""


def _score_from_errors(
    errors: list,
    weights: dict[str, float],
    cap: float = _PENALTY_CAP,
) -> float:
    """Compute a 0.0–1.0 score from a list of typed errors.

    Each error object must have a ``type`` attribute whose value is a key in
    *weights*. The score is ``max(0, 1 - sum(weights) / cap)``. An infinite
    weight floors the score at 0.0. An empty error list returns 1.0.

    Args:
        errors: The list of error objects from the judge (each with a
            ``type`` attribute).
        weights: Mapping from error type string to numeric weight.
        cap: The penalty sum at which the score reaches 0.0.

    Returns:
        A float in [0.0, 1.0].
    """
    if not errors:
        return 1.0
    total = 0.0
    for error in errors:
        weight = weights.get(getattr(error, 'type', ''), 1.0)
        if weight == float('inf'):
            return 0.0
        total += weight
    return max(0.0, 1.0 - total / cap)


def _judge(
    example: dspy.Example,
    prediction: dspy.Prediction,
    input_keys: list[str],
    expected_keys: list[str],
    predicted_key: str,
    judge_signature: type[dspy.Signature],
    judge_lm: Callable[[], dspy.LM],
    error_weights: dict[str, float],
    trace: object | None,
    **extra_kwargs,
) -> float | bool:
    """Run a discrete-error judge and derive a formulaic score.

    The judge always outputs an ``errors`` list (not a float ``score``).
    The score is computed from *error_weights* via ``_score_from_errors``.

    During optimisation (``trace is not None``) the score is binarised:
    any score above 0 becomes True (i.e. the example has fewer errors than
    the penalty cap). A score of 0 becomes False.

    Args:
        example: The gold example from the traces (inputs + expected
            outputs).
        prediction: What the optimiser's candidate produced.
        input_keys: The input field names to pass as ``inputs`` to the judge.
        expected_keys: The output field names to read from ``example`` as the
            expected / reference.
        predicted_key: The output field name to read from ``prediction``.
        judge_signature: The dspy.Signature subclass that defines the judge.
        judge_lm: Factory for the judge's LM (e.g. ``llm.metric_lm``).
        error_weights: Mapping from error type to weight. The judge must
            output an ``errors`` list whose items have a ``type`` attribute
            matching these keys.
        trace: Passed through from the metric; when not None we binarise.
        **extra_kwargs: Extra keyword fields to pass to the judge (e.g.
            ``page_image=...`` for the corrector VLM judge).

    Returns:
        A float score if trace is None, or a bool (True means the score
        is > 0) during optimisation.
    """
    serialised = _serialise(example, input_keys, expected_keys)
    predicted_text = _serialise_attr(prediction, predicted_key)
    if not predicted_text:
        return True if trace is not None else 1.0  # empty -> no penalty

    with dspy.context(lm=judge_lm()):
        judge = dspy.ChainOfThought(judge_signature)
        result = judge(
            inputs=serialised['inputs'],
            expected=serialised['expected'],
            predicted=predicted_text,
            **extra_kwargs,
        )
    errors = list(getattr(result, 'errors', []) or [])
    score = _score_from_errors(errors, error_weights, _PENALTY_CAP)
    if trace is not None:
        return score > 0
    return score


def _serialise(
    example: dspy.Example,
    input_keys: list[str],
    expected_keys: list[str],
) -> dict[str, str]:
    """Pull the input and expected fields from an example as display
    strings."""
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
