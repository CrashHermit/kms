"""Extractor-stage metric — structural node extraction."""

from __future__ import annotations

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in a structural extraction."""

    type: str = dspy.OutputField(
        desc='One of: missed_node, extra_node, wrong_type, '
             'wrong_boundary, wrong_order'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet or node index identifying the error.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing what is wrong and what the reference '
             'did instead.'
    )


class Judge(dspy.Signature):
    """You are evaluating a structural node extractor. It was asked to parse
    a textbook page's markdown into a flat list of top-level structural block
    nodes in document order.

    WHAT THE EXTRACTOR WAS ASKED TO DO — judge against this standard:

    This extractor is purely STRUCTURAL and domain-agnostic: emit only general
    document structure (paragraph, math, code, list, table, image, caption,
    header). Do NOT penalize it for failing to identify math-semantic units
    (definitions, theorems) — that is a different stage's job.

    The node type vocabulary is closed: paragraph, math, code, list, table,
    image, caption, header.

    Key rules the extractor had to follow:
    - One node per top-level markdown block. Do not break a block's sub-parts
      into separate nodes, and do not merge distinct blocks into one.
    - Segment on structure (block boundaries) only — never on meaning. Do NOT
      split a block because of what it says (e.g. a paragraph that runs into
      "Proof." stays one node).
    - Emit a whole list as a single list node — do not split it into per-item
      nodes.
    - Do not put caption text inside image nodes.
    - A short label that opens a labelled block (e.g. "Example 6.7",
      "Theorem 2.1") is a header.

    Your job is to FIND EVERY DISCRETE ERROR in the candidate's node list,
    categorized into these types:

    missed_node — A node that is in the reference but missing from the
      candidate.

    extra_node — A node in the candidate that has no counterpart in the
      reference.

    wrong_type — A node was extracted but given the wrong type (e.g. a header
      labelled as paragraph).

    wrong_boundary — A node's content boundaries are wrong: it either merges
      nodes that should be separate, or splits a single block into multiple
      nodes.

    wrong_order — Nodes are in the wrong document order.

    RULES:
    - Be exhaustive. List every error you find.
    - Content in the candidate that is verbatim from the reference but has
      different whitespace, trailing newlines, or equivalent markdown spacing
      is NOT an error.
    - If the candidate matches the reference in structure (same nodes, same
      types, same order, same content boundaries), return an empty error list.
    - Ignore the ``content`` field for type-comparison purposes — two nodes
      of the same type at the same position count as matching even if their
      content strings differ slightly in formatting."""

    inputs: str = dspy.InputField(
        desc='The segment markdown that was parsed '
             '(keyed as "segment_markdown: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference node list — what a correct extraction looks like.'
    )
    predicted: str = dspy.InputField(
        desc='The candidate node list — what the system actually produced.'
    )
    errors: list[Error] = dspy.OutputField(
        desc='Every discrete error found. Empty list if faithful to reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'missed_node': 2.0,
    'wrong_boundary': 2.0,
    'extra_node': 1.5,
    'wrong_type': 1.0,
    'wrong_order': 1.0,
}


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
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
