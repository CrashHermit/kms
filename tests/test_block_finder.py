"""Block finder: the cursor-walk banking rule (inherited verbatim from the per-type finders) and the
graph-node wrapper. The finder emits SPANS ONLY — the type is the attributor's job."""

import asyncio

from kms.core import models
from kms.entity.finders.block import BlockFinderNode, BlockSpan, find_blocks


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='intro prose',
            id=0,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.HEADER,
            content='Theorem 1',
            id=1,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='every n is prime',
            id=2,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='more prose',
            id=3,
            segment_index=0,
        ),
    ]


class _ScriptedFinder:
    """A stand-in Module whose aforward returns pre-scripted spans per call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_find_blocks_banks_a_bounded_block_and_emits_member_ids():
    # First read spans the theorem (local positions 1-2); node 3 follows it, so it is
    # bounded and banked. The cursor then advances past it and the tail read is empty.
    module = _ScriptedFinder([[BlockSpan(start=1, end=2)], []])
    blocks = asyncio.run(find_blocks(_nodes(), module=module))
    assert len(blocks) == 1
    assert blocks[0].members == [
        1,
        2,
    ]  # stable global ids, not window positions


def test_found_blocks_are_untyped_the_attributor_induces_the_type():
    module = _ScriptedFinder([[BlockSpan(start=1, end=2)], []])
    blocks = asyncio.run(find_blocks(_nodes(), module=module))
    assert blocks[0].type is None


def test_find_blocks_on_prose_only_stream_returns_nothing():
    module = _ScriptedFinder([[]])
    assert asyncio.run(find_blocks(_nodes(), module=module)) == []


def test_node_run_writes_the_block_channel():
    node = BlockFinderNode(
        module=_ScriptedFinder([[BlockSpan(start=1, end=2)], []])
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert list(out.keys()) == ['block_entities']
    assert [e.members for e in out['block_entities']] == [[1, 2]]


def test_node_run_on_empty_stream_yields_empty_channel():
    node = BlockFinderNode(module=_ScriptedFinder([]))
    assert asyncio.run(node.run({'nodes': []})) == {'block_entities': []}
