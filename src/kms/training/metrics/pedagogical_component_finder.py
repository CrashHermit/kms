"""Pedagogical-component-finder-stage metric — span-boundary detection."""

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in span-boundary detection."""

    type: str = dspy.OutputField(
        desc='One of: missed_span, extra_span, fused_spans, '
             'wrong_start, wrong_end'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet or position range identifying the error.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing the error.'
    )


class Judge(dspy.Signature):
    r"""You are evaluating a pedagogical component finder. It was asked to
    find the BOUNDARIES of every pedagogical unit in a run of textbook nodes
    and return each unit as a span of node positions.

    WHAT THE FINDER WAS ASKED TO DO — judge against this standard:

    This task is PURELY STRUCTURAL: say WHERE the units start and stop, never
    WHAT they are. Do not classify, label, or name them. Your only job is to
    cut the stream in the right places. FIND EVERYTHING: a missed unit is a
    deleted unit.

    A STATEMENT AND ITS DERIVATION ARE ALWAYS TWO SEPARATE SPANS. Never merge
    a theorem with its proof, or an example with its solution, into one span.

    NEVER SKIP A LABELLED UNIT. Every node that opens with its own label
    begins a span, without exception.

    A MARKER IS COMMON BUT NEVER REQUIRED for a derivation cut. Many books
    do not mark derivations — the absence of "Solution." is NOT a reason to
    keep statement and working in one span. Cut where the text stops posing
    and starts working.

    Your job is to find discrete errors, categorized as:

    missed_span — A span that is in the reference but missing from the
      candidate. This is the most severe error: a missed unit is a deleted
      unit.

    extra_span — A span in the candidate that has no counterpart in the
      reference. A false positive.

    fused_spans — The candidate merged two spans that the reference kept
      separate (e.g. did not cut between a statement and its derivation).
      A single fused span where two should exist.

    wrong_start — A span's start position is off by one or more nodes.

    wrong_end — A span's end position is off by one or more nodes.

    RULES:
    - Be exhaustive. List every error you find.
    - If the candidate's spans match the reference exactly (same positions,
      same boundaries), return an empty error list.
    - Off-by-one on a span boundary counts as wrong_start or wrong_end —
      these affect what content the downstream stages see."""

    inputs: str = dspy.InputField(
        desc='The window of nodes the finder saw '
             '(keyed as "current_nodes: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference spans — what a correct boundary cut looks like '
             '(keyed as "spans: ...").'
    )
    predicted: str = dspy.InputField(
        desc='The candidate spans — what the system produced '
             '(keyed as "spans: ...").'
    )
    errors: list[Error] = dspy.OutputField(
        desc='Every discrete error found. Empty list if faithful to reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'missed_span': 3.0,
    'fused_spans': 2.5,
    'extra_span': 2.0,
    'wrong_start': 1.0,
    'wrong_end': 1.0,
}


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
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
