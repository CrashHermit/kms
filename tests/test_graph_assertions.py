from kms.core import models
from kms.graph import assertions, entities, predicates
from kms.graph.triplets import triplet_uuid


def _triplet(
    subject,
    predicate,
    object,
    evidence_positions=None,
    *,
    subject_kind=models.NodeKind.ENTITY,
    object_kind=models.NodeKind.ENTITY,
):
    ids = evidence_positions or []
    return models.Triplet(
        subject=subject,
        predicate=predicate,
        object=object,
        subject_kind=subject_kind,
        object_kind=object_kind,
        evidence_positions=ids,
        occurrence_uuids={
            node_id: triplet_uuid(
                'hefferon.pdf', node_id, subject, predicate, object
            )
            for node_id in ids
        },
    )


def test_entity_uuid_is_local_per_node():
    at_three = entities.entity_uuid('hefferon.pdf', 3, '$G$ (a graph)')
    at_nine = entities.entity_uuid('hefferon.pdf', 9, '$G$ (a graph)')
    assert at_three != at_nine


def test_predicate_uuid_derives_from_triplet():
    first = predicates.predicate_uuid('triplet-a')
    second = predicates.predicate_uuid('triplet-b')
    assert first != second
    assert first == predicates.predicate_uuid('triplet-a')


def test_assertion_rows_builds_components_and_edges():
    triplet = _triplet(
        '$G_1$ (a graph)',
        'is a subgraph of',
        '$G_2$ (a graph)',
        evidence_positions=[3],
    )
    rows = assertions.assertion_rows(
        [triplet],
        'hefferon.pdf',
        {
            3: {
                '$G_1$ (a graph)': 'first graph',
                '$G_2$ (a graph)': 'second graph',
            }
        },
        {3: {'is a subgraph of': 'subset relation'}},
    )
    assert len(rows['entities']) == 2
    assert len(rows['predicates']) == 1
    assert len(rows['subject_edges']) == 1
    assert len(rows['object_edges']) == 1
    assert len(rows['predicate_edges']) == 1
    assert rows['subject_edges'][0]['entity'] == rows['entities'][0]['uuid']
    assert rows['object_edges'][0]['entity'] == rows['entities'][1]['uuid']
    assert len(rows['entity_names']) == 2
    assert len(rows['predicate_names']) == 1
    assert len(rows['entity_name_edges']) == 2
    assert len(rows['predicate_name_edges']) == 1

    by_name = {entity['name']: entity for entity in rows['entities']}
    assert by_name['$G_1$ (a graph)']['description'] == 'first graph'
    assert rows['predicates'][0]['description'] == 'subset relation'


def test_assertion_rows_collapse_shared_subject_and_object():
    triplet = _triplet('graph', 'equals', 'graph', evidence_positions=[3])
    rows = assertions.assertion_rows([triplet], 'hefferon.pdf', {}, {})
    assert len(rows['entities']) == 1


def test_assertion_rows_include_embeddings_when_provided():
    triplet = _triplet(
        '$G_1$ (a graph)',
        'is a subgraph of',
        '$G_2$ (a graph)',
        evidence_positions=[3],
    )
    rows = assertions.assertion_rows(
        [triplet],
        'hefferon.pdf',
        {
            3: {
                '$G_1$ (a graph)': 'first',
                '$G_2$ (a graph)': 'second',
            }
        },
        {3: {'is a subgraph of': 'subset'}},
        entity_embeddings={3: {'$G_1$ (a graph)': [0.1, 0.2]}},
        predicate_embeddings={3: {'is a subgraph of': [0.5, 0.6]}},
    )
    by_name = {entity['name']: entity for entity in rows['entities']}
    assert by_name['$G_1$ (a graph)']['embedding'] == [0.1, 0.2]
    assert 'embedding' not in by_name['$G_2$ (a graph)']
    assert rows['predicates'][0]['embedding'] == [0.5, 0.6]


def test_assertion_rows_builds_event_endpoints():
    triplet = _triplet(
        'Alice',
        'participates in',
        'an appointment',
        evidence_positions=[3],
        object_kind=models.NodeKind.EVENT,
    )
    rows = assertions.assertion_rows(
        [triplet],
        'hefferon.pdf',
        {3: {'Alice': 'entity person', 'an appointment': 'wrong entity text'}},
        {3: {'participates in': 'event participation'}},
        entity_embeddings={3: {'Alice': [0.1], 'an appointment': [0.2]}},
        event_descriptions={3: {'an appointment': 'event occurrence'}},
        event_embeddings={3: {'an appointment': [0.9]}},
    )
    assert len(rows['entities']) == 1
    assert rows['events'][0]['name'] == 'an appointment'
    assert rows['events'][0]['description'] == 'event occurrence'
    assert rows['events'][0]['embedding'] == [0.9]


def test_event_endpoint_uuid_is_disjoint_and_stable():
    event = assertions.events.event_uuid('hefferon.pdf', 3, 'appointment')
    entity = entities.entity_uuid('hefferon.pdf', 3, 'appointment')
    assert event == assertions.events.event_uuid(
        'hefferon.pdf', 3, 'appointment'
    )
    assert event != entity
