"""Splitter-stage metric — exercise-pack splitting decisions."""

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in an exercise-split decision."""

    type: str = dspy.OutputField(
        desc='One of: missed_split, wrong_split, wrong_boundary, '
             'wrong_number'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet or node position identifying the error.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing what is wrong and what the reference '
             'did.'
    )


class Judge(dspy.Signature):
    """You are evaluating an exercise splitter. It was asked to find nodes
    that pack TWO OR MORE numbered exercises into one block and split them
    into individual exercise pieces.

    WHAT THE SPLITTER WAS ASKED TO DO:

    Find any single node that packs two or more numbered exercises into one
    block (usually a list node). Return that node's position and its exercises
    in order, each with its own number and verbatim content. Preserve leading
    fragments and break out embedded lead-ins.

    A node holding only ONE exercise is NOT a split. Worked examples,
    definitions, theorems, prose, and headers are never splits.

    Your job is to find discrete errors, categorized as:

    missed_split — The reference split a packed node; the candidate did not
      split it at all. Those exercises remain fused.

    wrong_split — The candidate split a node that should NOT have been split
      (a single exercise, a worked example, prose, etc.).

    wrong_boundary — The candidate split a node at the wrong place — it
      produced the wrong number of pieces, or the split points are shifted
      relative to the reference.

    wrong_number — An exercise piece was given the wrong reference number.

    RULES:
    - Be exhaustive. List every error you find.
    - If the candidate's splits match the reference exactly (same positions
      split, same number of pieces per split, same numbers), return an empty
      error list.
    - Content that is verbatim from the reference but has minor whitespace
      differences is NOT an error."""

    inputs: str = dspy.InputField(
        desc='The window of nodes the splitter saw '
             '(keyed as "current_nodes: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference splits — which nodes were split and how.'
    )
    predicted: str = dspy.InputField(
        desc='The candidate splits — what the system actually produced.'
    )
    errors: list[Error] = dspy.OutputField(
        desc='Every discrete error found. Empty list if faithful to reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'missed_split': 2.5,
    'wrong_split': 2.0,
    'wrong_boundary': 1.5,
    'wrong_number': 1.0,
}


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
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
