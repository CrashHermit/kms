"""Role-typer-stage metric — statement/procedure classification."""

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in a statement/procedure classification."""

    type: str = dspy.OutputField(
        desc='One of: labelled_as_procedure, statement_as_procedure, '
             'procedure_as_statement'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet from the span that was misclassified.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing the error and what the correct role is.'
    )


class Judge(dspy.Signature):
    r"""You are evaluating a role classifier. It was asked to decide, for a
    single textbook text span, whether it is a STATEMENT (a block that poses
    or asserts something) or a PROCEDURE (a derivation that works something
    out).

    The only valid answers are "statement" or "procedure".

    WHAT THE CLASSIFIER WAS ASKED TO DO — judge against this standard:

    "statement" — a BLOCK. The text POSES or ASSERTS something. It is a
    declarative statement (definition, theorem, proposition, lemma, corollary,
    axiom, law, model, rule, principle), a posed task (worked example's
    statement, exercise for the reader). It says that something IS SO or what
    is to be DONE, without doing it.

    "procedure" — a DERIVATION. The text WORKS SOMETHING OUT: a proof, a
    solution, a derivation, a worked calculation. It substitutes, integrates,
    factors, splits into cases, applies a named result, computes, or concludes
    ("hence", "therefore", "so we get", "this completes the proof"). Working
    is not only algebra — exhibiting an answer, analyzing a case or figure,
    verifying or justifying all count as procedure.

    THE CRITICAL RULE: ITS OWN LABEL MAKES IT A BLOCK — CHECK THIS FIRST.
    If the text opens with a label naming it as a unit of the book
    ("Definition 2.5.1", "Theorem 3.4", "Example 6.7", "Lemma 1.2", or a
    bare exercise number like "12.", "949"), the answer is "statement",
    WHATEVER FOLLOWS THAT LABEL. A derivation never carries a block label of
    its own.

    A LEADING EXERCISE NUMBER COUNTS AS A LABEL EVEN WITH NO PUNCTUATION
    AFTER IT. "949 25 - 7" is an exercise (statement), not a computation
    (procedure) — the expression is the task itself and no result is shown.

    OTHERWISE, THE TEST IS WHAT THE TEXT DOES, NOT HOW IT IS LABELLED.
    A "Proof." or "Solution." marker means "procedure", but most derivations
    carry no marker at all. For unlabelled text, never answer "statement" only
    because a marker word is missing.

    Your job is to find discrete errors, categorized as:

    labelled_as_procedure — The most severe error. The span opens with its
      own block label (a definition, theorem, example, or exercise number),
      yet the candidate called it "procedure". A labelled block can never be
      a derivation.

    statement_as_procedure — The span is a statement (no label, but its
      content poses or asserts without working anything out), yet the
      candidate called it "procedure".

    procedure_as_statement — The span is clearly a derivation (working
      something out, with or without a "Proof." marker), yet the candidate
      called it "statement".

    RULES:
    - At most one error per span (a classification is either right or wrong).
    - If the candidate's role matches the reference exactly, return an empty
      error list.
    - If the candidate gave an answer other than "statement" or "procedure",
      treat it as wrong for the error type that best describes what the
      reference says."""

    inputs: str = dspy.InputField(
        desc='The text span content that was classified '
             '(keyed as "contents: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference role — "statement" or "procedure" '
             '(keyed as "role: ...").'
    )
    predicted: str = dspy.InputField(
        desc='The candidate role — what the system produced '
             '(keyed as "role: ...").'
    )
    errors: list[Error] = dspy.OutputField(
        desc='At most one error. Empty list if role matches reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'labelled_as_procedure': 4.0,
    'statement_as_procedure': 2.5,
    'procedure_as_statement': 1.5,
}


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
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
