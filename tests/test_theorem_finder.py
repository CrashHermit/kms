"""Theorem finder: core banking rule and the graph-node wrapper (channel + type)."""

import asyncio

from module.state import ASTNode, EntityType, NodeType
from module.theorem_finder import TheoremFinderNode, TheoremSpan, find_theorems


def _nodes():
    return [
        ASTNode(type=NodeType.HEADER, content="Theorem 3.2", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Every bounded sequence ...", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Proof. Suppose ...", id=2, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="ordinary prose", id=3, seg_index=0),
    ]


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def test_find_theorems_gathers_statement_and_proof_with_the_right_type():
    # The theorem spans its label, claim, and proof (0-2); node 3 follows, so it's bounded.
    module = _ScriptedFinder([[TheoremSpan(start=0, end=2)], []])
    theorems = asyncio.run(find_theorems(_nodes(), module=module))
    assert len(theorems) == 1
    assert theorems[0].type == EntityType.THEOREM
    assert theorems[0].members == [0, 1, 2]  # statement + proof nodes


def test_node_run_writes_the_theorem_channel():
    node = TheoremFinderNode(module=_ScriptedFinder([[TheoremSpan(start=0, end=2)], []]))
    out = asyncio.run(node.run({"nodes": _nodes()}))
    assert list(out.keys()) == ["theorem_entities"]
    assert [e.members for e in out["theorem_entities"]] == [[0, 1, 2]]
