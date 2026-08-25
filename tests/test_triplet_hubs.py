import asyncio

from kms.construction import triplet_hubs
from kms.core import identity, models
from kms.graph import hubs, queries, schema, writer


def test_triplet_hub_ids_are_tuple_stable_and_tier_scoped():
    source_id = hubs.triplet_hub_uuid(
        'source', 'book-a', 'subject', 'predicate', 'object'
    )
    assert source_id == hubs.triplet_hub_uuid(
        'source', 'book-a', 'subject', 'predicate', 'object'
    )
    assert source_id != hubs.triplet_hub_uuid(
        'source', 'book-b', 'subject', 'predicate', 'object'
    )
    assert source_id != hubs.triplet_hub_uuid(
        'meta', None, 'subject', 'predicate', 'object'
    )


def test_triplet_memberships_use_predicate_occurrence_identity():
    triplet = models.Triplet(
        subject='graph', predicate='has', object='vertex', evidence_positions=[0]
    )
    identity.assign_triplet_ids([triplet], 'book.pdf')
    subject_id = identity.entity_uuid('book.pdf', 0, triplet.subject)
    object_id = identity.entity_uuid('book.pdf', 0, triplet.object)
    predicate_id = identity.predicate_uuid(triplet.occurrence_uuids[0])

    memberships = triplet_hubs.build_triplet_memberships(
        [triplet],
        source='book.pdf',
        entity_assignments={
            subject_id: ('entity-hub',),
            object_id: ('object-hub',),
        },
        predicate_assignments={predicate_id: ('predicate-hub',)},
    )

    assert memberships[0].subject_hubs == ('entity-hub',)
    assert memberships[0].predicate_hubs == ('predicate-hub',)
    assert memberships[0].object_hubs == ('object-hub',)


def test_schema_contains_triplet_hub_constraints_and_indexes():
    combined = '\n'.join(schema.schema_statements())

    assert 'triplet_hub_uuid' in combined
    assert 'triplet_hub_embedding' in combined
    assert 'meta_triplet_hub_uuid' in combined
    assert 'meta_triplet_hub_embedding' in combined


def test_group_query_keeps_triplet_as_three_way_intersection_anchor():
    cypher = asyncio.run(_group_query_text('source'))
    assert 'MATCH (t:Triplet)' in cypher
    assert '(t)-[:HAS_SUBJECT]->(s:Entity)-[:CANONICAL]->' in cypher
    assert '(t)-[:HAS_PREDICATE]->(p:Predicate)-[:CANONICAL]->' in cypher
    assert '(t)-[:HAS_OBJECT]->(o:Entity)-[:CANONICAL]->' in cypher
    assert 'collect(DISTINCT t.uuid)' in cypher


def test_meta_group_query_requires_two_sources():
    cypher = asyncio.run(_group_query_text('meta'))
    assert 'ALIGNS_TO' in cypher
    assert 'WHERE size(sources) >= 2' in cypher
    assert 'local_tuples' in cypher


def test_prepare_meta_groups_maps_local_tuples_to_source_triplet_hubs():
    groups = triplet_hubs._prepare_groups(
        [
            {
                'subject_hub': 'meta-s',
                'subject_name': 'graph',
                'subject_description': 'A graph.',
                'predicate_hub': 'meta-p',
                'predicate_name': 'has',
                'predicate_description': 'Has.',
                'object_hub': 'meta-o',
                'object_name': 'vertex',
                'object_description': 'A vertex.',
                'sources': ['book-a', 'book-b'],
                'local_tuples': [
                    {
                        'source': 'book-a',
                        'subject_hub': 'source-s-a',
                        'predicate_hub': 'source-p-a',
                        'object_hub': 'source-o-a',
                    }
                ],
                'triplets': ['triplet-a'],
                'evidence': [],
            }
        ],
        'meta',
    )

    assert len(groups) == 1
    assert groups[0]['local_hubs'] == [
        hubs.triplet_hub_uuid(
            'source', 'book-a', 'source-s-a', 'source-p-a', 'source-o-a'
        )
    ]
    assert groups[0]['triplets'] == ['triplet-a']


def test_prepare_meta_groups_discards_underqualified_rows():
    assert (
        triplet_hubs._prepare_groups(
            [
                {
                    'subject_hub': 'meta-s',
                    'predicate_hub': 'meta-p',
                    'object_hub': 'meta-o',
                    'sources': ['book-a'],
                }
            ],
            'meta',
        )
        == []
    )


def test_persist_triplet_hubs_writes_roles_evidence_and_meta_support():
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))

    async def scenario():
        await writer.persist_triplet_hubs(
            [
                {
                    'uuid': 'meta-triplet',
                    'canonical_name': 'A graph has vertices.',
                    'description': 'A graph has vertices.',
                    'embedding': [0.1],
                    'subject_hub': 'meta-s',
                    'predicate_hub': 'meta-p',
                    'object_hub': 'meta-o',
                    'triplets': ['triplet-a'],
                    'local_hubs': ['triplet-hub-a', 'triplet-hub-b'],
                    'sources': ['book-a', 'book-b'],
                }
            ],
            tier='meta',
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert len(captured) == 3
    assert 'GlobalTripletHub' in captured[0][0]
    assert 'HAS_META_SUBJECT_HUB' in captured[1][0]
    assert captured[2][1]['pairs'] == [
        {'source_hub': 'triplet-hub-a', 'meta_hub': 'meta-triplet'},
        {'source_hub': 'triplet-hub-b', 'meta_hub': 'meta-triplet'},
    ]


def test_persist_meta_triplet_hubs_rejects_single_source_support():
    try:
        asyncio.run(
            writer.persist_triplet_hubs(
                [
                    {
                        'uuid': 'meta-triplet',
                        'canonical_name': 'A fact.',
                        'description': 'A fact.',
                        'subject_hub': 'meta-s',
                        'predicate_hub': 'meta-p',
                        'object_hub': 'meta-o',
                        'sources': ['book-a'],
                    }
                ],
                tier='meta',
                session_factory=lambda: None,
            )
        )
    except ValueError as error:
        assert 'two distinct source supports' in str(error)
    else:
        raise AssertionError('expected singleton meta support to be rejected')


async def _group_query_text(tier):
    captured = []

    class _Result:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append(cypher)
            return _Result()

    rows = await queries.triplet_hub_groups(lambda: _Session(), tier)
    assert rows == []
    return captured[0]
