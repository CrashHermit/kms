"""The graph persister stage — its skip gates, hermetically. The actual write path is covered by
the Neo4j integration test; here we only assert it no-ops (never touches the driver) when Neo4j
isn't configured or the run carries no source."""

import asyncio

from kms.core.models import ASTNode, NodeType
from kms.graph.persister import NodePersisterNode

_STATE = {
    "nodes": [ASTNode(type=NodeType.PARAGRAPH, content="x", id=0, seg_index=0)],
    "source": "book.pdf",
}


def test_no_op_when_neo4j_is_not_configured(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    assert asyncio.run(NodePersisterNode().run(_STATE)) == {}


def test_no_op_when_the_run_has_no_source(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")  # configured, but no source in state
    assert asyncio.run(NodePersisterNode().run({"nodes": _STATE["nodes"]})) == {}
