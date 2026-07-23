"""The graph persister stages — their skip gates, hermetically. The actual write path is covered by
the Neo4j integration test; here we only assert they no-op (never touch the driver) when Neo4j isn't
configured or the run carries no source."""

import asyncio

from kms.core.models import ASTNode, Entity, EntityType, NodeType
from kms.graph.persister import EntityPersisterNode, NodePersisterNode

_STATE = {
    "nodes": [ASTNode(type=NodeType.PARAGRAPH, content="x", id=0, seg_index=0)],
    "source": "book.pdf",
    "definition_entities": [Entity(type=EntityType.DEFINITION, members=[0])],
}


def test_node_persister_no_op_when_neo4j_is_not_configured(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    assert asyncio.run(NodePersisterNode().run(_STATE)) == {}


def test_node_persister_no_op_when_the_run_has_no_source(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")  # configured, but no source in state
    assert asyncio.run(NodePersisterNode().run({"nodes": _STATE["nodes"]})) == {}


def test_entity_persister_no_op_when_neo4j_is_not_configured(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    assert asyncio.run(EntityPersisterNode().run(_STATE)) == {}


def test_entity_persister_no_op_when_the_run_has_no_source(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")  # configured, but no source in state
    assert asyncio.run(EntityPersisterNode().run({"nodes": _STATE["nodes"]})) == {}
