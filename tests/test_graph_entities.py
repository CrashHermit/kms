"""Entity-layer graph mapping — the per-triplet ``:Entity`` vertices.

Pure mapping, asserted against deterministic uuids. The invariant this suite
guards: entity identity is PER-TRIPLET — every triplet's subject and object
get their own vertices, so two triplets at the same node sharing a surface
form NEVER share an entity (mirrors the per-triplet Predicate decision).
"""

from kms.core import models
from kms.graph import entities, nodes
from kms.graph.facts import fact_uuid
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


def _descriptions():
    return {
        3: [
            {'name': '$G_4$', 'description': 'The graph G4.'},
            {'name': '$G_1$', 'description': 'The graph G1.'},
        ],
        9: [
            {'name': '$G_4$', 'description': 'G4 again.'},
        ],
    }


def _t_uuid(source, node_id, fact, triplet):
    f_uuid = fact_uuid(source, fact.node_ids, triplet.fact_index)
    return triplet_uuid(
        source,
        node_id,
        f_uuid,
        triplet.subject,
        triplet.predicate,
        triplet.object,
    )


def test_entity_uuid_is_deterministic():
    t_uuid = _t_uuid(
        'hefferon.pdf', 3, _fact([3]), _triplet('$G_4$', 'is', '$G_1$')
    )
    assert entities.entity_uuid('hefferon.pdf', 3, t_uuid, 'subject') == (
        entities.entity_uuid('hefferon.pdf', 3, t_uuid, 'subject')
    )


def test_entity_identity_distinguishes_roles_within_a_triplet():
    t_uuid = _t_uuid(
        'hefferon.pdf', 3, _fact([3]), _triplet('$G_4$', 'is', '$G_1$')
    )
    assert entities.entity_uuid('hefferon.pdf', 3, t_uuid, 'subject') != (
        entities.entity_uuid('hefferon.pdf', 3, t_uuid, 'object')
    )


def test_entity_identity_is_per_triplet_not_shared():
    # Two triplets at the SAME node with the SAME subject get SEPARATE
    # vertices — zero sharing, matching the per-triplet Predicate decision.
    fact = _fact([3])
    first_triplet = _triplet('$G_4$', 'is', 'uncountable', fact_index=0)
    second_triplet = _triplet(
        '$G_4$', 'has cardinality', '$2^{\\aleph_0}$', fact_index=0
    )
    first = entities.entity_uuid(
        'hefferon.pdf',
        3,
        _t_uuid('hefferon.pdf', 3, fact, first_triplet),
        'subject',
    )
    second = entities.entity_uuid(
        'hefferon.pdf',
        3,
        _t_uuid('hefferon.pdf', 3, fact, second_triplet),
        'subject',
    )
    assert first != second


def test_entity_identity_is_local_per_node():
    fact = _fact([3, 9])
    triplet = _triplet('$G_4$', 'is', '$G_1$', fact_index=0)
    at_three = entities.entity_uuid(
        'hefferon.pdf',
        3,
        _t_uuid('hefferon.pdf', 3, fact, triplet),
        'subject',
    )
    at_nine = entities.entity_uuid(
        'hefferon.pdf',
        9,
        _t_uuid('hefferon.pdf', 9, fact, triplet),
        'subject',
    )
    assert at_three != at_nine


def test_entity_properties_carry_name_description_role_and_anchor():
    t_uuid = _t_uuid(
        'hefferon.pdf', 3, _fact([3]), _triplet('$G_4$', 'is', '$G_1$')
    )
    props = entities.entity_properties(
        'hefferon.pdf', 3, t_uuid, 'subject', '$G_4$', 'The graph.'
    )
    assert props['uuid'] == entities.entity_uuid(
        'hefferon.pdf', 3, t_uuid, 'subject'
    )
    assert props['source'] == nodes.source_uuid('hefferon.pdf')
    assert props['node_id'] == 3
    assert props['role'] == 'subject'
    assert props['name'] == '$G_4$'
    assert props['description'] == 'The graph.'


def test_entity_rows_one_row_per_triplet_role():
    facts = [_fact([3])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$G_1$', fact_index=0),
    ]
    rows = entities.entity_rows(
        triplets_list, facts, 'hefferon.pdf', _descriptions()
    )
    assert len(rows) == 2  # one subject + one object
    assert {row['role'] for row in rows} == {'subject', 'object'}


def test_entity_rows_duplicate_across_triplets_never_share():
    # Two triplets at the same node, same subject name -> TWO subject rows.
    facts = [_fact([3])]
    triplets_list = [
        _triplet('$G_4$', 'is', 'uncountable', fact_index=0),
        _triplet('$G_4$', 'has cardinality', '$2^{\\aleph_0}$', fact_index=0),
    ]
    rows = entities.entity_rows(
        triplets_list, facts, 'hefferon.pdf', _descriptions()
    )
    subjects = [row for row in rows if row['role'] == 'subject']
    assert len(subjects) == 2
    assert subjects[0]['uuid'] != subjects[1]['uuid']


def test_undescribed_endpoint_contributes_no_row():
    facts = [_fact([3])]
    triplets_list = [
        _triplet('$G_4$', 'is NOT a subgraph of', '$H_9$', fact_index=0),
    ]
    rows = entities.entity_rows(
        triplets_list, facts, 'hefferon.pdf', _descriptions()
    )
    roles = {row['role'] for row in rows}
    assert roles == {'subject'}  # object '$H_9$' was never described
