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


def test_statement_procedure_work_items_query_reads_raw_source():
    captured = {}

    async def scenario():
        await queries.statement_procedure_work_items(
            lambda: _capturing_session(captured)
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'src.key AS source' in cypher
    assert 'HAS_PROCEDURE' in cypher
    assert 'FIRST' in cypher


def test_compose_procedure_query_reads_members_in_order():
    captured = {}

    async def scenario():
        await queries.compose_procedure(
            'procedure-a', lambda: _capturing_session(captured)
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'MEMBER_OF' in cypher
    assert 'ORDER BY n.index' in cypher


def test_all_components_entity_returns_raw_source_key():
    captured = {}

    async def scenario():
        await queries.all_components(
            lambda: _capturing_session(captured), 'entity', source='book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'src.key AS source' in cypher
    assert 'c.name AS name' in cypher
    assert 't.source AS source' not in cypher


def test_all_components_predicate_reads_predicate_field():
    captured = {}

    async def scenario():
        await queries.all_components(
            lambda: _capturing_session(captured),
            'predicate',
            source='book-a',
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'src.key AS source' in cypher
    assert 'c.predicate AS name' in cypher


def test_unassigned_components_exclude_canonicalized_records():
    captured = {}

    async def scenario():
        await queries.unassigned_components(
            lambda: _capturing_session(captured), 'entity', 'book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'NOT (c)-[:CANONICAL]->(:EntityHub)' in cypher
    assert 'src.key = $source' in cypher


def test_all_source_hubs_reads_canonical_fields_and_can_filter():
    captured = {}

    async def scenario():
        await queries.all_source_hubs(
            lambda: _capturing_session(captured), 'entity', source='book-a'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'MATCH (h:EntityHub)' in cypher
    assert 'h.canonical_name AS name' in cypher
    assert 'h.aliases AS aliases' in cypher
    assert 'src.key AS source' in cypher
    assert 'WHERE h.source = $source_uuid' in cypher


def test_all_source_hubs_can_filter_to_changed_hub_ids():
    captured = {}

    async def scenario():
        await queries.all_source_hubs(
            lambda: _capturing_session(captured),
            'entity',
            hub_uuids=['hub-a', 'hub-b'],
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'WHERE h.uuid IN $hub_uuids' in cypher


def test_qualified_meta_hub_query_requires_two_source_values():
    captured = {}

    async def scenario():
        await queries.qualified_meta_hub_uuids(
            lambda: _capturing_session(captured), 'entity'
        )

    asyncio.run(scenario())

    cypher = '\n'.join(captured['queries'])
    assert 'OPTIONAL MATCH (m)<-[:ALIGNS_TO]-(s:EntityHub)' in cypher
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
        await queries.all_components(
            lambda: _Session(), 'entity', source='book-a'
        )

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

    assert 'MATCH (node:TripletHub)' in calls[0][0]
    assert calls[1][1]['k'] == 12
