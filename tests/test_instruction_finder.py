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
        models.ASTNode(type='image', image_path='a.png', id=1),
        models.ASTNode(type='list', content='3 matrix A', id=2),
        models.ASTNode(type='list', content='4 matrix B', id=3),
        models.ASTNode(type='paragraph', content='ordinary prose', id=4),
    ]


def test_finder_node_emits_instruction_hubs_without_mutating_nodes():
    span = instruction_finder.Span(start=0, end=1)
    node = instruction_finder.InstructionFinderNode(
        module=_ScriptedFinder([[span]])
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))

    assert set(out) == {'instructions'}
    assert len(out['instructions']) == 1
    instruction = out['instructions'][0]
    assert instruction.block == [0, 1]
    assert instruction.members == [0, 1]


def test_finder_node_emits_nothing_without_spans():
    node = instruction_finder.InstructionFinderNode(
        module=_ScriptedFinder([[]])
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert out['instructions'] == []


def test_find_instruction_spans_maps_positions_to_node_ids():
    spans = [
        instruction_finder.Span(start=0, end=1),
        instruction_finder.Span(start=2, end=2),
    ]
    result = asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes(), module=_ScriptedFinder([spans])
        )
    )
    assert result == [[0, 1], [2]]
