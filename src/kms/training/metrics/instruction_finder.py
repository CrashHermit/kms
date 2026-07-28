"""Instruction-finder-stage metric — lead-in position detection."""

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in lead-in detection."""

    type: str = dspy.OutputField(
        desc='One of: missed_lead_in, false_lead_in'
    )
    location: str = dspy.OutputField(
        desc='The node position or a short quoted snippet.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing the error.'
    )


class Judge(dspy.Signature):
    r"""You are evaluating an instruction finder. It was asked to identify
    exercise LEAD-IN nodes in a run of textbook nodes and return their
    positions.

    WHAT THE FINDER WAS ASKED TO DO:

    A lead-in is a short directive that introduces a run of OTHER exercises
    and states a shared instruction. The decisive test: a lead-in has NO
    reference number of its own, yet gives an imperative meant for a run of
    separately-numbered exercises that follow it.

    CRITICAL: a node that BEGINS WITH ITS OWN EXERCISE NUMBER ("1.15 Perform
    each multiplication.", "1.22 Represent each linear map ...") is an
    EXERCISE, never a lead-in — do NOT expect the candidate to tag it.

    Range-naming and range-less forms are BOTH valid lead-ins.

    Your job is to find discrete errors, categorized as:

    missed_lead_in — The reference tagged a lead-in; the candidate did not.
      This is the most severe error: a missed lead-in silently drops a shared
      instruction from all its governed exercises.

    false_lead_in — The candidate tagged a node as a lead-in that is not one,
      or tagged a self-numbered exercise as a lead-in. False positives on
      self-numbered exercises are the most common confusion.

    RULES:
    - Be exhaustive. List every error you find.
    - An empty position list is a valid answer if there truly are no lead-ins
      in the window.
    - If the candidate's positions match the reference exactly, return an
      empty error list.
    - Position order does not matter — only set membership."""

    inputs: str = dspy.InputField(
        desc='The window of nodes the finder saw '
             '(keyed as "current_nodes: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference lead-in positions '
             '(keyed as "instruction_positions: ...").'
    )
    predicted: str = dspy.InputField(
        desc='The candidate lead-in positions — what the system produced.'
    )
    errors: list[Error] = dspy.OutputField(
        desc='Every discrete error found. Empty list if faithful to reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'missed_lead_in': 3.0,
    'false_lead_in': 1.5,
}


def instruction_finder_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge whether the instruction finder tagged the correct lead-in
    positions."""
    return _judge(
        example,
        prediction,
        input_keys=['current_nodes'],
        expected_keys=['instruction_positions'],
        predicted_key='instruction_positions',
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
