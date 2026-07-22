"""Exercise governor: banking a grouped-exercise block and splitting it into per-exercise
Problem entities with a shared instruction. The LLM is injected via a scripted module."""

import asyncio

from module.state import ASTNode, NodeType, EntityType
from module.exercise_governor import govern_exercises, ExerciseBlock, ExerciseItem


def _nodes():
    return [
        ASTNode(type=NodeType.PARAGRAPH, content="In Exercises 1.23-1.24, find the eigenvalues.", id=0, seg_index=0),
        ASTNode(type=NodeType.LIST, content="1.23 $A=[[1,4],[4,1]]$\n1.24 $B=[[2,0],[0,3]]$", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="ordinary prose after", id=2, seg_index=0),
    ]


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_block_splits_into_one_entity_per_exercise_with_shared_instruction():
    block = ExerciseBlock(
        start=0, end=1,
        instruction="find the eigenvalues",
        exercises=[
            ExerciseItem(number="1.23", content="$A=[[1,4],[4,1]]$"),
            ExerciseItem(number="1.24", content="$B=[[2,0],[0,3]]$"),
        ],
    )
    entities = asyncio.run(govern_exercises(_nodes(), module=_ScriptedFinder([[block], []])))

    assert len(entities) == 2
    assert all(e.type == EntityType.PROBLEM for e in entities)
    # Distinct contents/number, shared instruction, shared coarse provenance (the block nodes).
    assert [e.number for e in entities] == ["1.23", "1.24"]
    assert [e.contents for e in entities] == [["$A=[[1,4],[4,1]]$"], ["$B=[[2,0],[0,3]]$"]]
    assert all(e.instruction == "find the eigenvalues" for e in entities)
    assert all(e.members == [0, 1] for e in entities)


def test_no_lead_in_leaves_instruction_none_but_still_splits():
    block = ExerciseBlock(
        start=0, end=0,
        instruction="",
        exercises=[ExerciseItem(number="5", content="Prove X."), ExerciseItem(number="6", content="Prove Y.")],
    )
    nodes = [ASTNode(type=NodeType.LIST, content="5 Prove X.\n6 Prove Y.", id=0, seg_index=0),
             ASTNode(type=NodeType.PARAGRAPH, content="after", id=1, seg_index=0)]
    entities = asyncio.run(govern_exercises(nodes, module=_ScriptedFinder([[block], []])))

    assert len(entities) == 2
    assert all(e.instruction is None for e in entities)
    assert [e.number for e in entities] == ["5", "6"]


def test_no_blocks_emits_nothing():
    entities = asyncio.run(govern_exercises(_nodes(), module=_ScriptedFinder([[], []])))
    assert entities == []
