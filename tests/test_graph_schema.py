"""Structural-node schema DDL — pure statement generation, no database (neo4j is stubbed in
conftest)."""

from kms.graph.schema import schema_statements


def test_declares_uuid_constraints_for_node_source_and_entity():
    stmts = schema_statements()
    assert any("(n:Node)" in s and "IS UNIQUE" in s for s in stmts)
    assert any("(s:Source)" in s and "IS UNIQUE" in s for s in stmts)
    assert any("(e:Entity)" in s and "IS UNIQUE" in s for s in stmts)


def test_no_separate_canonical_constraint_since_entity_covers_it():
    # reference canonicals are :Entity, so the entity uuid constraint already keys them
    assert not any("Canonical" in s or "GeneralEntity" in s for s in schema_statements())


def test_declares_uuid_constraints_for_the_procedural_and_concept_layers():
    stmts = schema_statements()
    assert any("(p:Procedure)" in s and "IS UNIQUE" in s for s in stmts)
    assert any("(v:Event)" in s and "IS UNIQUE" in s for s in stmts)
    assert any("(c:Concept)" in s and "IS UNIQUE" in s for s in stmts)


def test_indexes_node_and_entity_source_for_book_scoped_lookups():
    stmts = schema_statements()
    assert any("INDEX" in s and "ON (n.source)" in s for s in stmts)
    assert any("INDEX" in s and "ON (e.source)" in s for s in stmts)


def test_every_statement_is_idempotent():
    assert all("IF NOT EXISTS" in s for s in schema_statements())


def test_no_type_index_since_per_type_labels_supersede_it():
    # kind lookups are native label scans (:Math, …), so no property index is created
    assert not any("ON (n.type)" in s for s in schema_statements())


def test_no_vector_index_at_the_structural_layer():
    assert not any("VECTOR" in s for s in schema_statements())
