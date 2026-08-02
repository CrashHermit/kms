"""Instruction finder: tags exercise lead-in nodes."""

import asyncio

from kms.core import models
from kms.ingestion import instruction_finder


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def _nodes():
    return [
        models.ASTNode(
            type='paragraph',
            content='In the following exercises, simplify.',
            id=0,
        ),
        models.ASTNode(type='list', content='3 matrix A', id=1),
        models.ASTNode(type='list', content='4 matrix B', id=2),
        models.ASTNode(type='paragraph', content='ordinary prose', id=3),
    ]


def test_tags_the_lead_in_node_and_leaves_others_untouched():
    out = asyncio.run(
        instruction_finder.tag_instructions(
            _nodes(), module=_ScriptedFinder([[0]])
        )
    )
    assert out[0].type == 'instruction'
    assert out[1].type == 'list'
    assert out[2].type == 'list'
    assert out[3].type == 'paragraph'


def test_no_lead_in_leaves_every_node_unchanged():
    out = asyncio.run(
        instruction_finder.tag_instructions(
            _nodes(), module=_ScriptedFinder([[]])
        )
    )
    assert all(not n.type == 'instruction' for n in out)


def test_out_of_range_position_is_clamped_not_fatal():
    out = asyncio.run(
        instruction_finder.tag_instructions(
            _nodes(), module=_ScriptedFinder([[99]])
        )
    )
    assert out[3].type == 'instruction'


def test_instruction_finder_node_writes_the_nodes_channel():
    node = instruction_finder.InstructionFinderNode(
        module=_ScriptedFinder([[0]])
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'nodes'}
    assert out['nodes'][0].type == 'instruction'
