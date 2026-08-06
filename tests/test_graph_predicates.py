"""Predicate-layer graph mapping — the described predicate component.

Each predicate vertex is per (node, triplet): it carries the predicate text
and the enricher's description, and hangs off its triplet hub via
:HAS_PREDICATE. This suite guards the per-triplet identity (same predicate
text in two triplets of one fact is deliberately duplicated, never shared).
"""

from kms.core import models
from kms.graph import entities, nodes
from kms.graph.facts import fact_uuid
from kms.graph.predicates import (
    has_predicate_pairs,
    predicate_properties,
    predicate_rows,
    predicate_uuid,
)
from kms.graph.triplets import triplet_uuid


def _fact(node_ids):
    return models.AtomicFact(text='x', node_ids=node_ids)


def _triplet(subject, predicate, object, fact_index=0):
    return models.Triplet(
        subject=subject,
        predicate=predicate,
        object=object,
        fact_index=fact_index,
    )


def _relation_descriptions():
    return {
        3: [
            {
                'predicate': 'is NOT a subgraph of',
                'description': 'G4 is not a subgraph of G1.',
            }
        ],
        9: [
            {
                'predicate': 'is NOT a subgraph of',
                'description': 'G4 is not a subgraph of G1 (later).',
            }
        ],
    }


def test_predicate_uuid_is_deterministic_and_disjoint():
    fact = _fact([3])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    first = predicate_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    second = predicate_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert first == second
    assert first != f_uuid
    triplet = triplet_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert first != entities.entity_uuid('hefferon.pdf', 3, triplet, 'subject')
    assert first != triplet


def test_predicate_is_duplicated_per_triplet_not_shared():
    # Two triplets of the SAME fact sharing a predicate text still get their
    # OWN predicate vertices — the description is deliberately duplicated.
    fact = _fact([3])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    first = predicate_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is', 'uncountable'
    )
    second = predicate_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is', 'connected'
    )
    assert first != second


def test_predicate_uuid_distinguishes_per_node_occurrences():
    fact = _fact([3, 9])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    at_three = predicate_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    at_nine = predicate_uuid(
        'hefferon.pdf', 9, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert at_three != at_nine


def test_predicate_properties_carry_text_description_and_anchor():
    fact = _fact([3])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    props = predicate_properties(
        'hefferon.pdf',
        3,
        f_uuid,
        '$G_4$',
        'is NOT a subgraph of',
        '$G_1$',
        'G4 is not a subgraph of G1.',
    )
    assert props['uuid'] == predicate_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert props['source'] == nodes.source_uuid('hefferon.pdf')
    assert props['node_id'] == 3
    assert props['predicate'] == 'is NOT a subgraph of'
    assert props['description'] == 'G4 is not a subgraph of G1.'


def test_predicate_rows_write_each_node_occurrence_separately():
    # A fact touching two nodes produces TWO rows, each carrying that node's
    # OWN description — never collapsed to the first node.
    facts = [_fact([3, 9])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', fact_index=0),
    ]
    rows = predicate_rows(
        triplets_list, facts, 'hefferon.pdf', _relation_descriptions()
    )
    assert len(rows) == 2
    descriptions = {row['node_id']: row['description'] for row in rows}
    assert descriptions[3] == 'G4 is not a subgraph of G1.'
    assert descriptions[9] == 'G4 is not a subgraph of G1 (later).'


def test_predicate_rows_omit_missing_description():
    facts = [_fact([3])]
    triplets_list = [_triplet('$G_4$', 'is', '$G_1$', fact_index=0)]
    rows = predicate_rows(triplets_list, facts, 'hefferon.pdf', {})
    assert len(rows) == 1
    assert 'description' not in rows[0]
    assert rows[0]['node_id'] == 3


def test_has_predicate_pairs_link_hub_to_component():
    facts = [_fact([3, 9])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', fact_index=0),
    ]
    pairs = has_predicate_pairs(triplets_list, facts, 'hefferon.pdf')
    assert len(pairs) == 2
    f_uuid = fact_uuid('hefferon.pdf', [3, 9], 0)
    for pair in pairs:
        # The hub and the component are distinct, but the predicate pair
        # names the same per-node occurrence as the triplet hub does.
        assert pair['predicate'] != pair['triplet']
        assert pair['triplet'] in {
            triplet_uuid(
                'hefferon.pdf',
                node_id,
                f_uuid,
                '$G_4$',
                'is NOT a subgraph of',
                '$G_1$',
            )
            for node_id in (3, 9)
        }
