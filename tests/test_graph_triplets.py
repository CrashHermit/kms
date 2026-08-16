from kms.core import models
from kms.graph import nodes
from kms.graph.triplets import (
    evidence_pairs,
    triplet_properties,
    triplet_rows,
    triplet_uuid,
)


def _triplet(subject, predicate, object, node_ids=None):
    return models.Triplet(
        subject=subject,
        predicate=predicate,
        object=object,
        node_ids=node_ids or [],
    )


def test_triplet_uuid_is_deterministic_and_disjoint():
    first = triplet_uuid(
        'hefferon.pdf', 3, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    second = triplet_uuid(
        'hefferon.pdf', 3, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert first == second
    different = triplet_uuid('hefferon.pdf', 3, '$G_4$', 'is', '$G_1$')
    assert first != different


def test_triplet_uuid_distinguishes_per_node_occurrences():
    at_three = triplet_uuid(
        'hefferon.pdf', 3, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    at_nine = triplet_uuid(
        'hefferon.pdf', 9, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert at_three != at_nine


def test_triplet_uuid_changes_when_the_content_changes():
    first = triplet_uuid(
        'hefferon.pdf', 1, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    second = triplet_uuid(
        'hefferon.pdf', 1, '$G_4$', 'is a subgraph of', '$G_1$'
    )
    assert first != second


def test_triplet_properties_are_a_pure_connector():
    triplet = _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', node_ids=[3])
    props = triplet_properties(triplet, 'hefferon.pdf', 3)
    assert props['uuid'] == triplet_uuid(
        'hefferon.pdf', 3, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert props['source'] == nodes.source_uuid('hefferon.pdf')
    assert props['node_id'] == 3
    assert 'subject' not in props
    assert 'predicate' not in props
    assert 'object' not in props


def test_triplet_rows_write_each_node_occurrence_separately():
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', node_ids=[3, 9]),
    ]
    rows = triplet_rows(triplets_list, 'hefferon.pdf')
    assert len(rows) == 2
    assert {row['node_id'] for row in rows} == {3, 9}


def test_evidence_pairs_one_pair_per_node_occurrence():
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', node_ids=[3, 9]),
    ]
    pairs = evidence_pairs(triplets_list, 'hefferon.pdf')
    assert len(pairs) == 2
    expected_triplet_uuid = triplet_uuid(
        'hefferon.pdf', 3, '$G_4$', 'is NOT a subgraph of', '$G_1$'
    )
    assert {
        'node': nodes.node_uuid('hefferon.pdf', 3),
        'triplet': expected_triplet_uuid,
    } in pairs
    assert {
        'node': nodes.node_uuid('hefferon.pdf', 9),
        'triplet': triplet_uuid(
            'hefferon.pdf', 9, '$G_4$', 'is NOT a subgraph of', '$G_1$'
        ),
    } in pairs
