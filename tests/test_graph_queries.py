import asyncio

from kms.graph import queries


def _capturing_session(captured):
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
            captured.setdefault('queries', []).append(cypher)
            return _Result()

    return _Session()


def test_all_components_entity_returns_raw_source_key():
    captured = {}

    async def scenario():
        await queries.all_entity_components(
            lambda: _capturing_session(captured), source='book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'src.key AS source' in cypher
    assert 'c.name AS name' in cypher
    assert 'c.node_position AS node_position' in cypher
    assert 'c.node_id' not in cypher
    assert 't.source AS source' not in cypher


def test_all_components_predicate_reads_predicate_field():
    captured = {}

    async def scenario():
        await queries.all_predicate_components(
            lambda: _capturing_session(captured), source='book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'src.key AS source' in cypher
    assert 'c.predicate AS name' in cypher


def test_all_components_event_reads_event_name_and_label():
    captured = {}

    async def scenario():
        await queries.all_event_components(
            lambda: _capturing_session(captured), source='book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'MATCH (c:Event)' in cypher
    assert 'c.name AS name' in cypher
    assert 'WHERE src.key = $source' in cypher


def test_unassigned_components_exclude_canonicalized_records():
    captured = {}

    async def scenario():
        await queries.unassigned_entity_components(
            lambda: _capturing_session(captured), 'book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'NOT (c)-[:CANONICAL]->(:LocalEntityHub)' in cypher
    assert 'src.key = $source' in cypher


def test_all_source_hubs_reads_canonical_fields_and_can_filter():
    captured = {}

    async def scenario():
        await queries.all_entity_source_hubs(
            lambda: _capturing_session(captured), source='book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'MATCH (h:LocalEntityHub)' in cypher
    assert 'h.canonical_name AS name' in cypher
    assert 'h.aliases AS aliases' in cypher
    assert 'src.key AS source' in cypher
    assert 'WHERE h.source = $source_uuid' in cypher


def test_all_source_hubs_can_filter_to_changed_hub_ids():
    captured = {}

    async def scenario():
        await queries.all_entity_source_hubs(
            lambda: _capturing_session(captured),
            hub_uuids=['hub-a', 'hub-b'],
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'WHERE h.uuid IN $hub_uuids' in cypher


def test_qualified_meta_hub_query_requires_two_source_values():
    captured = {}

    async def scenario():
        await queries.qualified_entity_meta_hub_uuids(
            lambda: _capturing_session(captured)
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'OPTIONAL MATCH (m)<-[:ALIGNS_TO]-(s:LocalEntityHub)' in cypher
    assert 'count(DISTINCT s.source)' in cypher
    assert 'WHERE source_count >= 2' in cypher


def test_all_components_can_filter_to_one_source():
    captured = {}

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
            captured['cypher'] = cypher
            captured['kwargs'] = kwargs
            return _Result()

    async def scenario():
        await queries.all_entity_components(lambda: _Session(), source='book-a')

    asyncio.run(scenario())

    assert 'WHERE src.key = $source' in captured['cypher']
    assert captured['kwargs'] == {'source': 'book-a'}


def test_vector_search_overfetches_known_source_indexes():
    calls = []

    class _Result:
        def __init__(self, record=None):
            self.record = record

        async def single(self):
            return self.record

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
            calls.append((cypher, kwargs))
            if len(calls) == 1:
                return _Result({'count': 12})
            return _Result()

    async def scenario():
        await queries.vector_search(
            lambda: _Session(),
            index_name='triplet_hub_embedding',
            query_embedding=[0.1],
            top_k=3,
            source='book-a',
        )

    asyncio.run(scenario())

    assert 'MATCH (node:LocalTripletHub)' in calls[0][0]
    assert calls[1][1]['k'] == 12
