import asyncio

import pytest

from kms.postprocessing.learning import event_cards


def test_event_fact_indexes_are_unique_positive():
    with pytest.raises(ValueError):
        event_cards.EventCardFact(text='x', evidence_indexes=[1, 1])


def test_event_card_creation_uses_event_evidence():
    class Router:
        async def aforward(self, **kwargs): return True
    class Facts:
        async def aforward(self, **kwargs):
            return [event_cards.EventCardFact(text='The event occurred.', evidence_indexes=[1])]
    class Generator:
        async def aforward(self, **kwargs):
            return [event_cards.EventCardDraft(prompt='What occurred?', response='The event occurred.', fact_index=1)]
    class Verifier:
        async def aforward(self, **kwargs):
            assert 'Battle: Source evidence.' in kwargs['evidence']
            return event_cards.EventCardVerification(supported=True)
    candidate = event_cards.EventCardCandidateInput(
        global_hub={'uuid': 'g'}, local_hub={'name': 'Battle', 'local_uuid': 'h'},
        source='book', evidence=[{'index': 1, 'uuid': 'event-1', 'name': 'Battle', 'description': 'Source evidence.'}],
    )
    cards = asyncio.run(event_cards.create_event_cards(candidate, router=Router(), fact_generator=Facts(), card_generator=Generator(), verifier=Verifier()))
    assert cards[0].evidence_uuids == ('event-1',)
