"""Structural-node schema DDL — pure statement generation, no database (neo4j is stubbed in
conftest)."""

from kms.graph.schema import schema_statements


def test_declares_the_uuid_uniqueness_constraint():
    stmts = schema_statements()
    assert any("CONSTRAINT" in s and "IS UNIQUE" in s for s in stmts)


def test_statements_target_the_node_label():
    assert all("(n:Node)" in s for s in schema_statements())


def test_every_statement_is_idempotent():
    assert all("IF NOT EXISTS" in s for s in schema_statements())


def test_no_type_index_since_per_type_labels_supersede_it():
    # kind lookups are native label scans (:Math, …), so no property index is created
    assert not any("ON (n.type)" in s for s in schema_statements())


def test_no_vector_index_at_the_structural_layer():
    assert not any("VECTOR" in s for s in schema_statements())
