"""Seam-merger-stage metric — cross-page boundary merge decisions."""

from __future__ import annotations

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


class Error(dspy.Signature):
    """One discrete error in a seam merge decision."""

    type: str = dspy.OutputField(
        desc='One of: false_merge, false_split'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet from the candidate showing the error.'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing what the candidate did wrong and what '
             'the reference did.'
    )


class Judge(dspy.Signature):
    """You are evaluating a seam-merger decision for a cross-page boundary.
    The merger was given two edge nodes (the tail of one page and the head of
    the next) plus their context neighbours, and asked whether they are two
    halves of the same interrupted block.

    WHAT THE SEAM MERGER WAS ASKED TO DO:

    Decide whether the tail node of the top run and the head node of the
    bottom run are two halves of one split block. If they are, merge them
    into one coherent node. If they are already complete, independent nodes
    that merely sit next to each other at the boundary, return None.

    Judge this purely on STRUCTURE: does the tail read as cut off mid-block
    and the head as its continuation? Use the context nodes only to inform
    judgment — never include their content in the merged output.

    Your job is to find discrete errors, categorized as:

    false_merge — The candidate merged two nodes that should have been
      left as separate, independent blocks. This is the most severe error:
      gluing unrelated text together corrupts both blocks.

    false_split — The candidate left two halves separate when they should
      have been merged into one node. This is recoverable downstream but
      leaves the document fragmented.

    RULES:
    - At most one error per decision (a merge is either right or wrong).
    - If the candidate made the same decision as the reference (both merged
      with equivalent content, or both returned None), return an empty error
      list.
    - A merge whose content is equivalent to the reference's merge (same
      meaning, even if slightly different wording or whitespace) is correct
      — do NOT flag it.
    - Do NOT flag cosmetic differences in the merged content if the decision
      to merge was correct."""

    inputs: str = dspy.InputField(
        desc='The four context nodes the merger saw '
             '(top_node_context, top_bottom_edge_node, '
             'bottom_top_edge_node, bottom_node_context).'
    )
    expected: str = dspy.InputField(
        desc='The reference merge decision (a merged node or None).'
    )
    predicted: str = dspy.InputField(
        desc='The candidate merge decision (a merged node or None).'
    )
    errors: list[Error] = dspy.OutputField(
        desc='At most one error. Empty list if decision matches reference.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary, or "no errors" if clean.'
    )


ERROR_WEIGHTS: dict[str, float] = {
    'false_merge': 5.0,
    'false_split': 1.0,
}


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
        judge_signature=Judge,
        judge_lm=llm.metric_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
    )
