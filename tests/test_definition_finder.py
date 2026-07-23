"""Definition finder: core banking rule and the graph-node wrapper (channel + type)."""

import asyncio

from kms.core.models import ASTNode, EntityType, NodeType
from kms.entity.finders.definition import DefinitionFinderNode, DefinitionSpan, find_definitions


def _nodes():
    return [
        ASTNode(type=NodeType.HEADER, content="Definition 2.1", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="A vector space is ...", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="ordinary prose", id=2, seg_index=0),
    ]


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_find_definitions_banks_a_bounded_definition_with_the_right_type():
    module = _ScriptedFinder([[DefinitionSpan(start=0, end=1)], []])
    definitions = asyncio.run(find_definitions(_nodes(), module=module))
    assert len(definitions) == 1
    assert definitions[0].type == EntityType.DEFINITION
    assert definitions[0].members == [0, 1]


def test_node_run_writes_the_definition_channel():
    node = DefinitionFinderNode(module=_ScriptedFinder([[DefinitionSpan(start=0, end=1)], []]))
    out = asyncio.run(node.run({"nodes": _nodes()}))
    assert list(out.keys()) == ["definition_entities"]
    assert [e.members for e in out["definition_entities"]] == [[0, 1]]
