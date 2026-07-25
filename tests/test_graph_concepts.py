"""Concept hub identity — the naming and dedup scheme the (currently dark) concept layer will
write through. Pure, no database.

The layer emits nothing today: its only source was the deleted ``field`` taxonomy, and
conceptualization is not built yet (see ``docs/CONCEPT-LAYER.md``). What is tested here is the
half that survives — global, source-free identity so the same concept converges on one node
across books, and cheap exact-name clustering."""

from kms.graph.concepts import (
    concept_batch,
    concept_properties,
    concept_uuid,
    normalize_concept,
)
from kms.graph.entities import entity_uuid
from kms.graph.nodes import node_uuid

# --- normalization ---


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_concept('  Linear   Algebra ') == 'linear algebra'
    assert normalize_concept('Chain Rule') == normalize_concept('chain  rule')


def test_normalization_does_not_merge_genuine_paraphrases():
    # only spacing/case variants converge; "vector spaces" vs "linear algebra" is the
    # fusion tier's job, not this one
    assert normalize_concept('vector spaces') != normalize_concept(
        'linear algebra'
    )


# --- identity ---


def test_concept_uuid_is_deterministic_and_name_keyed():
    assert concept_uuid('chain rule') == concept_uuid('chain rule')
    assert concept_uuid('chain rule') != concept_uuid('product rule')


def test_concept_uuid_is_global_so_books_converge_on_one_node():
    # no source prefix anywhere in the key: the same concept induced from any book is one node
    assert concept_uuid('Chain Rule') == concept_uuid('  chain   rule  ')


def test_concept_uuid_is_disjoint_from_the_other_namespaces():
    assert concept_uuid('1') != entity_uuid('1', 1)
    assert concept_uuid('1') != node_uuid('1', 1)


# --- properties and batching ---


def test_concept_properties_carry_uuid_and_name_but_no_source():
    props = concept_properties('  Chain Rule ')
    assert props == {'uuid': concept_uuid('chain rule'), 'name': 'Chain Rule'}
    assert 'source' not in props  # a concept is corpus-level, not book-scoped


def test_concept_batch_dedupes_by_uuid_and_drops_blanks():
    batch = concept_batch(['Chain Rule', 'chain  rule', '', '  ', 'Limits'])
    assert len(batch) == 2
    assert {props['uuid'] for props in batch} == {
        concept_uuid('chain rule'),
        concept_uuid('limits'),
    }
