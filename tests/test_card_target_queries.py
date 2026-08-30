import asyncio

import pytest

from kms.core import models
from kms.graph import queries


class _Result:
    def __init__(self, rows):
        self.rows = rows

    async def single(self):
        return self.rows[0] if self.rows else None

    def __aiter__(self):
        self.iterator = iter(self.rows)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _Result(self.rows)


class _Factory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return None


def test_hub_context_uses_local_and_global_typed_labels():
    session = _Session(
        [
            {
                'local_uuid': 'local',
                'local_name': 'Local',
                'local_description': 'd',
                'global_uuid': None,
                'global_name': None,
                'global_description': None,
                'source': 'book',
            }
        ]
    )
    result = asyncio.run(
        queries.card_hub_context(
            _Factory(session),
            target_kind=models.CardTargetKind.TRIPLET,
            hub_uuid='local',
        )
    )
    assert result['global_uuid'] is None
    assert 'LocalTripletHub' in session.calls[0][0]
    assert 'GlobalTripletHub' in session.calls[0][0]


def test_missing_hub_raises_descriptive_error():
    with pytest.raises(ValueError, match='card hub not found: missing'):
        asyncio.run(
            queries.card_hub_context(
                _Factory(_Session([])),
                target_kind=models.CardTargetKind.ENTITY,
                hub_uuid='missing',
            )
        )


def test_triplet_targets_reconstruct_endpoints():
    session = _Session(
        [
            {
                'uuid': 'triplet',
                'kind': 'triplet',
                'source': 'book',
                'content': 'A --not causes--> B',
                'context': {},
            }
        ]
    )
    targets = asyncio.run(
        queries.card_targets_for_hub(
            _Factory(session),
            target_kind=models.CardTargetKind.TRIPLET,
            hub_uuid='hub',
        )
    )
    assert targets[0]['uuid'] == 'triplet'
    cypher = session.calls[0][0]
    for relationship in (
        'CANONICAL',
        'HAS_SUBJECT',
        'HAS_PREDICATE',
        'HAS_OBJECT',
    ):
        assert relationship in cypher
    assert 'predicate.predicate' in cypher
    for obsolete in (
        'INSTANCE_OF',
        'HAS_SUBJECT_HUB',
        'HAS_PREDICATE_HUB',
        'HAS_OBJECT_HUB',
    ):
        assert obsolete not in cypher


def test_empty_entity_content_is_excluded():
    session = _Session(
        [
            {
                'uuid': 'entity',
                'kind': 'entity',
                'source': 'book',
                'content': ' ',
                'context': {},
            }
        ]
    )
    targets = asyncio.run(
        queries.card_targets_for_hub(
            _Factory(session),
            target_kind=models.CardTargetKind.ENTITY,
            hub_uuid='hub',
        )
    )
    assert targets == []
