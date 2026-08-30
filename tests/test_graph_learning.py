import asyncio

from kms.core import models
from kms.graph import writer


class _Result:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Session:
    def __init__(self):
        self.calls = []

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return _Result()


class _Factory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return None


def test_persist_card_target_edges_emits_component_pairs():
    session = _Session()
    cards = [
        models.Card('entity-card', 'entity', models.CardTargetKind.ENTITY),
        models.Card('event-card', 'event', models.CardTargetKind.EVENT),
        models.Card('triplet-card', 'triplet', models.CardTargetKind.TRIPLET),
        models.Card(
            'procedure-card', 'procedure', models.CardTargetKind.PROCEDURE
        ),
    ]
    asyncio.run(
        writer.persist_card_target_edges(
            cards, session_factory=_Factory(session)
        )
    )
    query, kwargs = session.calls[0]
    assert 'HAS_CARD' in query
    assert 'SUPPORTS' not in query
    assert kwargs['pairs'] == [
        {'target': 'entity', 'card': 'entity-card'},
        {'target': 'event', 'card': 'event-card'},
        {'target': 'triplet', 'card': 'triplet-card'},
        {'target': 'procedure', 'card': 'procedure-card'},
    ]
