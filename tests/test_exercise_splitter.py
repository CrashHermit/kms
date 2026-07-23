"""Exercise splitter: split packed exercise nodes into per-exercise nodes and tag lead-ins.
The LLM is injected via a scripted module returning (splits, instruction_positions) per window."""

import asyncio

from kms.core.models import ASTNode, NodeType
from kms.entity.splitter import NodeSplit, SplitExercise, SplitterNode, split_exercises


class _ScriptedSplitter:
    """Replays one (splits, instruction_positions) verdict per window call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else ([], [])


def _nodes():
    return [
        ASTNode(
            type=NodeType.PARAGRAPH,
            content="In Exercises 3-4, compute the determinant.",
            id=0,
            seg_index=0,
        ),
        ASTNode(type=NodeType.LIST, content="3 matrix A\n4 matrix B", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="ordinary prose", id=2, seg_index=0),
    ]


def test_splits_a_packed_node_and_tags_the_lead_in():
    # One window: node 1 splits into two exercises; node 0 is the lead-in.
    split = NodeSplit(
        position=1,
        exercises=[
            SplitExercise(number="3", content="matrix A"),
            SplitExercise(number="4", content="matrix B"),
        ],
    )
    out = asyncio.run(split_exercises(_nodes(), module=_ScriptedSplitter([([split], [0])])))

    # 3 nodes -> 4 (the packed node became two), ids re-assigned 0..3, seg_index inherited.
    assert [n.id for n in out] == [0, 1, 2, 3]
    assert [n.content for n in out] == [
        "In Exercises 3-4, compute the determinant.",
        "3 matrix A",
        "4 matrix B",
        "ordinary prose",
    ]
    assert all(n.seg_index == 0 for n in out)
    # The lead-in is tagged; the split pieces and prose are not.
    assert out[0].role == "instruction"
    assert [n.role for n in out[1:]] == [None, None, None]
    # A split piece inherits the parent node's structural type.
    assert out[1].type == NodeType.LIST and out[2].type == NodeType.LIST


def test_an_embedded_lead_in_piece_is_tagged():
    # A packed node whose middle piece is a lead-in (instruction=True) is tagged in place.
    split = NodeSplit(
        position=1,
        exercises=[
            SplitExercise(number="3", content="matrix A"),
            SplitExercise(number="", content="4-5 find the inverse.", instruction=True),
            SplitExercise(number="4", content="matrix B"),
        ],
    )
    out = asyncio.run(split_exercises(_nodes(), module=_ScriptedSplitter([([split], [])])))
    assert [n.content for n in out] == [
        "In Exercises 3-4, compute the determinant.",
        "3 matrix A",
        "4-5 find the inverse.",
        "4 matrix B",
        "ordinary prose",
    ]
    # Only the embedded lead-in piece carries the instruction role; the exercises don't.
    assert [n.role for n in out] == [None, None, "instruction", None, None]


def test_single_exercise_is_not_split():
    # A verdict with only one exercise must be ignored (only GROUPS split).
    split = NodeSplit(position=1, exercises=[SplitExercise(number="3", content="only one")])
    out = asyncio.run(split_exercises(_nodes(), module=_ScriptedSplitter([([split], [])])))
    assert len(out) == 3  # unchanged
    assert [n.content for n in out] == [n.content for n in _nodes()]


def test_no_verdict_passes_the_stream_through_unchanged():
    out = asyncio.run(split_exercises(_nodes(), module=_ScriptedSplitter([([], [])])))
    assert [(n.id, n.content, n.role) for n in out] == [
        (0, "In Exercises 3-4, compute the determinant.", None),
        (1, "3 matrix A\n4 matrix B", None),
        (2, "ordinary prose", None),
    ]


def test_splitter_node_writes_the_nodes_channel():
    split = NodeSplit(
        position=1,
        exercises=[
            SplitExercise(number="3", content="matrix A"),
            SplitExercise(number="4", content="matrix B"),
        ],
    )
    node = SplitterNode(module=_ScriptedSplitter([([split], [0])]))
    out = asyncio.run(node.run({"nodes": _nodes()}))
    assert set(out) == {"nodes"}
    assert len(out["nodes"]) == 4
