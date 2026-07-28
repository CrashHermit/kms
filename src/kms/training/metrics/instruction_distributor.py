"""Instruction-distributor-stage metric — governance-range assignment."""

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in governance-range assignment."""

    type: str = dspy.OutputField(
        desc='One of: missed_exercise, extra_exercise, wrong_boundary, '
             'wrong_instruction'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet or position identifying the error.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing the error.'
    )


class Judge(dspy.Signature):
    """You are evaluating an instruction distributor. It was given an exercise
    lead-in and the exercise nodes that follow it, and asked to decide which
    exercises the lead-in's shared instruction governs.

    WHAT THE DISTRIBUTOR WAS ASKED TO DO:

    Judge governance by meaning, not by numbers: the governed exercises are a
    contiguous run that starts at the first exercise after the lead-in and
    continues while the lead-in's instruction still sensibly applies, stopping
    when it no longer does.

    Return the shared instruction (without range framing) and the positions of
    the governed exercises.

    Your job is to find discrete errors, categorized as:

    missed_exercise — An exercise that should be governed (per the reference)
      but was not included. It will miss the shared instruction.

    extra_exercise — An exercise that was included in the governed set but
      should not be. It will get an instruction that does not apply.

    wrong_boundary — The governed range is wrong at one or both edges: it
      starts too late or ends too early/late, even if some exercises in the
      middle are correct.

    wrong_instruction — The extracted instruction text differs from the
      reference (wrong content, still includes range framing, or empty when
      it should not be).

    RULES:
    - Be exhaustive. List every error you find.
    - If the candidate's governed positions and instruction match the
      reference exactly, return an empty error list.
    - Minor wording differences in the instruction that preserve meaning
      are NOT errors."""

    inputs: str = dspy.InputField(
        desc='The lead-in text and following problems the distributor saw '
             '(keyed as "lead_in: ..." and "following_problems: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference instruction and governed positions '
             '(keyed as "instruction: ..." and "governed_positions: ...").'
    )
    predicted: str = dspy.InputField(
        desc='The candidate governed positions — what the system produced '
             '(keyed as "governed_positions: ...").'
    )
    errors: list[Error] = dspy.OutputField(
        desc='Every discrete error found. Empty list if faithful to reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'missed_exercise': 2.0,
    'wrong_boundary': 2.0,
    'extra_exercise': 1.5,
    'wrong_instruction': 1.5,
}


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
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
