"""Instruction finder: tags exercise lead-in nodes with type=INSTRUCTION.
The LLM is injected via a scripted module returning the lead-in positions per window."""

import asyncio

from kms.core import models
from kms.ingestion.instruction_finder import (
    InstructionFinderNode,
    tag_instructions,
)


class _ScriptedFinder:
    """Replays one `instruction_positions` verdict per window call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='In the following exercises, simplify.',
            id=0,
        ),
        models.ASTNode(type=models.NodeType.LIST, content='3 matrix A', id=1),
        models.ASTNode(type=models.NodeType.LIST, content='4 matrix B', id=2),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content='ordinary prose', id=3
        ),
    ]


def test_tags_the_lead_in_node_and_leaves_others_untouched():
    # One window; position 0 is the lead-in.
    out = asyncio.run(
        tag_instructions(_nodes(), module=_ScriptedFinder([[0]]))
    )
    assert [n.type for n in out] == [
        models.NodeType.INSTRUCTION,
        models.NodeType.LIST,
        models.NodeType.LIST,
        models.NodeType.PARAGRAPH,
    ]
    # Content is never touched — tagging is a pure type change.
    assert out[0].content == 'In the following exercises, simplify.'


def test_no_lead_in_leaves_every_node_unchanged():
    out = asyncio.run(
        tag_instructions(_nodes(), module=_ScriptedFinder([[]]))
    )
    assert all(n.type != models.NodeType.INSTRUCTION for n in out)


def test_out_of_range_position_is_clamped_not_fatal():
    # A stray position past the window edge clamps to the last node rather than
    # crashing.
    out = asyncio.run(
        tag_instructions(_nodes(), module=_ScriptedFinder([[99]]))
    )
    assert [n.type for n in out] == [
        models.NodeType.PARAGRAPH,
        models.NodeType.LIST,
        models.NodeType.LIST,
        models.NodeType.INSTRUCTION,
    ]


def test_instruction_finder_node_writes_the_nodes_channel():
    node = InstructionFinderNode(module=_ScriptedFinder([[0]]))
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'nodes'}
    assert out['nodes'][0].type == models.NodeType.INSTRUCTION
