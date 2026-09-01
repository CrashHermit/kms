import asyncio

from kms.construction import triplet_hubs
from kms.core import models
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


def test_schema_contains_triplet_hub_constraints_and_indexes():
    combined = '\n'.join(schema.schema_statements())

    assert 'triplet_hub_uuid' in combined
    assert 'triplet_hub_embedding' in combined
    assert 'global_triplet_hub_uuid' in combined
    assert 'global_triplet_hub_embedding' in combined


def test_group_query_keeps_triplet_as_three_way_intersection_anchor():
    cypher = asyncio.run(_group_query_text('source'))
    assert 'MATCH (t:Triplet)' in cypher
    assert '(t)-[:HAS_SUBJECT]->(s)-[:CANONICAL]->(sh)' in cypher
    assert '(t)-[:HAS_PREDICATE]->(p:Predicate)-[:CANONICAL]->' in cypher
    assert '(t)-[:HAS_OBJECT]->(o)-[:CANONICAL]->(oh)' in cypher
    assert f'sh:{hubs.LOCAL_EVENT_HUB_LABEL}' in cypher
    assert f'oh:{hubs.ENTITY_HUB_LABEL}' in cypher
    assert f'oh:{hubs.LOCAL_EVENT_HUB_LABEL}' in cypher


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
    assert 'HAS_GLOBAL_SUBJECT_HUB' in captured[1][0]
    assert captured[2][1]['pairs'] == [
        {'source_hub': 'triplet-hub-a', 'meta_hub': 'meta-triplet'},
        {'source_hub': 'triplet-hub-b', 'meta_hub': 'meta-triplet'},
    ]


def test_persist_global_triplet_hubs_rejects_single_source_support():
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


def test_source_role_query_accepts_entity_or_event_hubs():
    cypher = queries.merge_triplet_hub_edges_query('source')

    assert '(s {uuid: row.subject_hub})' in cypher
    assert f's:{hubs.ENTITY_HUB_LABEL}' in cypher
    assert f's:{hubs.LOCAL_EVENT_HUB_LABEL}' in cypher
    assert f'o:{hubs.ENTITY_HUB_LABEL}' in cypher
    assert f'o:{hubs.LOCAL_EVENT_HUB_LABEL}' in cypher
    assert f'p:{hubs.PREDICATE_HUB_LABEL}' in cypher
    assert 'HAS_SUBJECT_HUB' in cypher
    assert 'HAS_PREDICATE_HUB' in cypher
    assert 'HAS_OBJECT_HUB' in cypher


def test_persist_source_triplet_hub_writes_roles_and_evidence():
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))

    asyncio.run(
        writer.persist_triplet_hubs(
            [
                {
                    'uuid': 'source-triplet',
                    'source': 'book-a',
                    'canonical_name': 'An event precedes an entity.',
                    'description': 'An event precedes an entity.',
                    'embedding': [0.1],
                    'subject_hub': 'event-hub',
                    'predicate_hub': 'predicate-hub',
                    'object_hub': 'entity-hub',
                    'triplets': ['triplet-a'],
                }
            ],
            tier='source',
            session_factory=lambda: _Session(),
        )
    )

    assert len(captured) == 3
    assert 'LocalEventHub' in captured[1][0]
    assert captured[1][1]['rows'] == [
        {
            'hub': 'source-triplet',
            'subject_hub': 'event-hub',
            'predicate_hub': 'predicate-hub',
            'object_hub': 'entity-hub',
        }
    ]
    assert captured[2][1]['pairs'] == [
        {'triplet': 'triplet-a', 'hub': 'source-triplet'}
    ]
    assert 'HAS_SUBJECT_HUB' in captured[1][0]
    assert 'HAS_PREDICATE_HUB' in captured[1][0]
    assert 'HAS_OBJECT_HUB' in captured[1][0]
    assert 'CANONICAL' in captured[2][0]


def test_rebuild_source_triplet_hubs_clears_before_persisting(monkeypatch):
    calls = []
    rows = [
        {
            'source': 'book-a',
            'subject_hub': 'event-hub',
            'predicate_hub': 'predicate-hub',
            'object_hub': 'entity-hub',
            'triplets': ['triplet-a'],
        }
    ]
    groups = [
        {
            **rows[0],
            'uuid': 'source-triplet',
            'canonical_name': 'An event precedes an entity.',
            'description': 'An event precedes an entity.',
            'embedding': [0.1],
        }
    ]

    async def read_groups(*args, **kwargs):
        calls.append('read')
        return rows

    async def synthesize(groups_to_synthesize, **kwargs):
        calls.append('synthesize')
        assert groups_to_synthesize == triplet_hubs._prepare_groups(
            rows, 'source'
        )
        return groups

    async def clear(*args, **kwargs):
        calls.append(('clear', kwargs['source']))

    async def persist(groups_to_persist, **kwargs):
        calls.append(('persist', groups_to_persist))

    monkeypatch.setattr(triplet_hubs.queries, 'triplet_hub_groups', read_groups)
    monkeypatch.setattr(triplet_hubs, '_synthesize_groups', synthesize)
    monkeypatch.setattr(triplet_hubs.writer, 'clear_triplet_hubs', clear)
    monkeypatch.setattr(triplet_hubs.writer, 'persist_triplet_hubs', persist)

    result = asyncio.run(
        triplet_hubs.rebuild(
            language_model=object(),
            session_factory=object(),
            source='book-a',
            max_concurrency=4,
        )
    )

    assert calls[0:2] == ['read', 'synthesize']
    assert calls[2][0] == 'clear'
    assert calls[3][0] == 'persist'
    assert calls[2][1] == 'book-a'
    assert result == {'triplet_hubs': 1, 'triplets': 1}


def test_rebuild_source_triplet_hubs_clears_when_no_groups(monkeypatch):
    calls = []

    async def read_groups(*args, **kwargs):
        return []

    async def clear(*args, **kwargs):
        calls.append(('clear', kwargs['source']))

    async def persist(groups, **kwargs):
        calls.append(('persist', groups))

    monkeypatch.setattr(triplet_hubs.queries, 'triplet_hub_groups', read_groups)
    monkeypatch.setattr(triplet_hubs.writer, 'clear_triplet_hubs', clear)
    monkeypatch.setattr(triplet_hubs.writer, 'persist_triplet_hubs', persist)

    result = asyncio.run(
        triplet_hubs.rebuild(
            language_model=object(),
            session_factory=object(),
            source='book-a',
        )
    )

    assert calls == [('clear', 'book-a'), ('persist', [])]
    assert result == {'triplet_hubs': 0, 'triplets': 0}


def test_triplet_hub_node_skips_without_graph(monkeypatch):
    async def fail_rebuild(**kwargs):
        raise AssertionError('rebuild must not run without graph persistence')

    monkeypatch.setattr(triplet_hubs, 'rebuild', fail_rebuild)
    node = triplet_hubs.TripletHubNode(
        object(),
        session_factory=None,
        neo4j_configured=False,
    )

    result = asyncio.run(node.run({'source': models.Source(key='book-a')}))

    assert result == {
        'triplet_hubs_created': 0,
        'triplets_clustered': 0,
    }


def test_triplet_hub_node_rebuilds_for_current_source(monkeypatch):
    captured = {}

    async def fake_rebuild(**kwargs):
        captured.update(kwargs)
        return {'triplet_hubs': 3, 'triplets': 7}

    monkeypatch.setattr(triplet_hubs, 'rebuild', fake_rebuild)
    language_model = object()
    session_factory = object()
    node = triplet_hubs.TripletHubNode(
        language_model,
        session_factory=session_factory,
        neo4j_configured=True,
        max_concurrency=4,
    )

    result = asyncio.run(node.run({'source': models.Source(key='book-a')}))

    assert result == {
        'triplet_hubs_created': 3,
        'triplets_clustered': 7,
    }
    assert captured == {
        'language_model': language_model,
        'session_factory': session_factory,
        'source': 'book-a',
        'max_concurrency': 4,
    }
