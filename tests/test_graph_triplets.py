"""Triplet-layer graph mapping — ``models.Triplet`` → ``:Triplet`` hubs.

Each triplet is a verbatim hub carrying its subject/predicate/object strings,
written separately per node occurrence. This suite guards the identity scheme
(anchored on the source fact AND the anchor node) and the role edges to
node-local entities.
"""

from kms.core import models
from kms.graph import entities, nodes
from kms.graph.facts import fact_uuid
from kms.graph.triplets import (
    has_object_pairs,
    has_subject_pairs,
    triplet_properties,
    triplet_rows,
    triplet_uuid,
    yields_pairs,
)


def _fact(node_ids):
    return models.AtomicFact(text='x', node_ids=node_ids)


def _triplet(subject, predicate, object, fact_index=0):
    return models.Triplet(
        subject=subject,
        predicate=predicate,
        object=object,
        fact_index=fact_index,
    )


def _entity_descriptions():
    return {
        3: [
            {'name': '$G_4$', 'description': 'The graph G4.'},
            {'name': '$G_1$', 'description': 'The graph G1.'},
        ],
        9: [
            {'name': '$G_4$', 'description': 'G4 again.'},
        ],
    }


def test_triplet_uuid_is_deterministic_and_disjoint():
    fact = _fact([3])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    first = triplet_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    second = triplet_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert first == second
    assert first != f_uuid
    assert first != entities.entity_uuid('hefferon.pdf', 3, first, 'subject')


def test_triplet_uuid_distinguishes_per_node_occurrences():
    # The same triplet whose fact touches two nodes is TWO hubs — each node
    # gets its own occurrence with its own entity endpoints.
    fact = _fact([3, 9])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    at_three = triplet_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    at_nine = triplet_uuid(
        'hefferon.pdf', 9, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert at_three != at_nine


def test_triplet_uuid_changes_when_the_source_fact_changes():
    old_fact = _fact([1])
    new_fact = _fact([2])
    triplet = _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$')
    old_uuid = triplet_uuid(
        'hefferon.pdf',
        1,
        fact_uuid('hefferon.pdf', old_fact.node_ids, 4),
        triplet.subject,
        triplet.predicate,
        triplet.object,
    )
    new_uuid = triplet_uuid(
        'hefferon.pdf',
        2,
        fact_uuid('hefferon.pdf', new_fact.node_ids, 4),
        triplet.subject,
        triplet.predicate,
        triplet.object,
    )
    assert old_uuid != new_uuid


def test_triplet_properties_carry_the_verbatim_strings():
    fact = _fact([3])
    f_uuid = fact_uuid('hefferon.pdf', fact.node_ids, 0)
    triplet = _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$')
    props = triplet_properties(triplet, 'hefferon.pdf', 3, f_uuid)
    assert props['uuid'] == triplet_uuid(
        'hefferon.pdf', 3, f_uuid, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert props['source'] == nodes.source_uuid('hefferon.pdf')
    assert props['node_id'] == 3
    assert props['subject'] == '$G_4$'
    assert props['predicate'] == 'is NOT a subgraph of'
    assert props['object'] == '$G_1$'
    assert props['fact_index'] == 0


def test_triplet_rows_write_each_node_occurrence_separately():
    facts = [_fact([3, 9])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', fact_index=0),
    ]
    rows = triplet_rows(triplets_list, facts, 'hefferon.pdf')
    assert len(rows) == 2
    assert {row['node_id'] for row in rows} == {3, 9}


def test_yields_pairs_one_pair_per_node_occurrence():
    facts = [_fact([3, 9])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', fact_index=0),
    ]
    pairs = yields_pairs(triplets_list, facts, 'hefferon.pdf')
    assert len(pairs) == 2
    assert {pair['fact'] for pair in pairs} == {
        fact_uuid('hefferon.pdf', [3, 9], 0)
    }


def test_has_subject_and_has_object_point_at_the_local_entities():
    facts = [_fact([3])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', fact_index=0),
    ]
    subjects = has_subject_pairs(
        triplets_list, facts, 'hefferon.pdf', _entity_descriptions()
    )
    objects = has_object_pairs(
        triplets_list, facts, 'hefferon.pdf', _entity_descriptions()
    )
    expected_triplet = triplet_uuid(
        'hefferon.pdf',
        3,
        fact_uuid('hefferon.pdf', [3], 0),
        '$G_4$',
        'is NOT a subgraph of',
        '$G_1$',
    )
    assert subjects == [
        {
            'triplet': expected_triplet,
            'entity': entities.entity_uuid(
                'hefferon.pdf', 3, expected_triplet, 'subject'
            ),
        }
    ]
    assert objects == [
        {
            'triplet': expected_triplet,
            'entity': entities.entity_uuid(
                'hefferon.pdf', 3, expected_triplet, 'object'
            ),
        }
    ]


def test_undescribed_endpoint_contributes_no_edge():
    # An entity the enricher never described gets no HAS_SUBJECT/HAS_OBJECT
    # edge — the verbatim string survives on the triplet hub instead.
    facts = [_fact([3])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$H_9$', fact_index=0),
    ]
    subjects = has_subject_pairs(
        triplets_list, facts, 'hefferon.pdf', _entity_descriptions()
    )
    objects = has_object_pairs(
        triplets_list, facts, 'hefferon.pdf', _entity_descriptions()
    )
    assert len(subjects) == 1
    assert objects == []
