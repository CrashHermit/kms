"""Exercise splitter: split packed exercise nodes into per-exercise nodes (split only — tagging
lead-ins is the instruction finder's job). The LLM is injected via a scripted module returning
the splits per window."""

import asyncio

from kms.core import models
from kms.ingestion.splitter import (
    NodeSplit,
    SplitExercise,
    SplitterNode,
    split_exercises,
)


class _ScriptedSplitter:
    """Replays one `splits` verdict per window call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='In Exercises 3-4, compute the determinant.',
            id=0,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.LIST,
            content='3 matrix A\n4 matrix B',
            id=1,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='ordinary prose',
            id=2,
            segment_index=0,
        ),
    ]


def test_splits_a_packed_node():
    # One window: node 1 splits into two exercises.
    split = NodeSplit(
        position=1,
        exercises=[
            SplitExercise(number='3', content='matrix A'),
            SplitExercise(number='4', content='matrix B'),
        ],
    )
    out = asyncio.run(
        split_exercises(_nodes(), module=_ScriptedSplitter([[split]]))
    )

    # 3 nodes -> 4 (the packed node became two), ids re-assigned 0..3, segment_index inherited.
    assert [n.id for n in out] == [0, 1, 2, 3]
    assert [n.content for n in out] == [
        'In Exercises 3-4, compute the determinant.',
        '3 matrix A',
        '4 matrix B',
        'ordinary prose',
    ]
    assert all(n.segment_index == 0 for n in out)
    # A split piece inherits the parent node's structural type.
    assert (
        out[1].type == models.NodeType.LIST
        and out[2].type == models.NodeType.LIST
    )


def test_an_embedded_lead_in_is_broken_out_as_its_own_node():
    # A packed node whose middle piece is a lead-in (empty number) lands on its own node.
    split = NodeSplit(
        position=1,
        exercises=[
            SplitExercise(number='3', content='matrix A'),
            SplitExercise(number='', content='4-5 find the inverse.'),
            SplitExercise(number='4', content='matrix B'),
        ],
    )
    out = asyncio.run(
        split_exercises(_nodes(), module=_ScriptedSplitter([[split]]))
    )
    assert [n.content for n in out] == [
        'In Exercises 3-4, compute the determinant.',
        '3 matrix A',
        '4-5 find the inverse.',
        '4 matrix B',
        'ordinary prose',
    ]


def test_single_exercise_is_not_split():
    # A verdict with only one exercise must be ignored (only GROUPS split).
    split = NodeSplit(
        position=1, exercises=[SplitExercise(number='3', content='only one')]
    )
    out = asyncio.run(
        split_exercises(_nodes(), module=_ScriptedSplitter([[split]]))
    )
    assert len(out) == 3  # unchanged
    assert [n.content for n in out] == [n.content for n in _nodes()]


def test_no_verdict_passes_the_stream_through_unchanged():
    out = asyncio.run(split_exercises(_nodes(), module=_ScriptedSplitter([[]])))
    assert [(n.id, n.content) for n in out] == [
        (0, 'In Exercises 3-4, compute the determinant.'),
        (1, '3 matrix A\n4 matrix B'),
        (2, 'ordinary prose'),
    ]


def test_splitter_node_writes_the_nodes_channel():
    split = NodeSplit(
        position=1,
        exercises=[
            SplitExercise(number='3', content='matrix A'),
            SplitExercise(number='4', content='matrix B'),
        ],
    )
    node = SplitterNode(module=_ScriptedSplitter([[split]]))
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'nodes'}
    assert len(out['nodes']) == 4
