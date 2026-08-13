import asyncio

from kms.graph import queries


def test_all_entity_spokes_returns_raw_source_key():
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
            return _Result()

    async def scenario():
        await queries.all_entity_spokes(lambda: _Session())

    asyncio.run(scenario())

    assert 'src.key AS source' in captured['cypher']
    assert 't.source AS source' not in captured['cypher']
