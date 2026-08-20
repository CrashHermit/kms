import asyncio
from types import SimpleNamespace

from kms.construction import entity_cards


class _Router:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FactGenerator:
    def __init__(self, facts):
        self.facts = facts
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return self.facts


class _CardGenerator:
    def __init__(self):
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        fact = kwargs['fact']
        return [
            entity_cards.EntityCardDraft(
                prompt='What property does it have?',
                response=fact,
            ),
        ]


def test_entity_card_router_decodes_boolean_prediction():
    prediction = SimpleNamespace(has_testable_knowledge=True)
    assert entity_cards.EntityCardRouter.decode(None, prediction) is True


def test_entity_fact_generator_strips_empty_facts():
    prediction = SimpleNamespace(facts=[' Fact one. ', '', 'Fact two. '])
    facts = entity_cards.EntityFactGenerator.decode(
        None, prediction, canonical_name='Group', description='description'
    )
    assert facts == ['Fact one.', 'Fact two.']


def test_entity_card_generator_preserves_fact():
    fact = 'A group has an identity element.'
    prediction = SimpleNamespace(
        card={
            'prompt': 'What element must a group have?',
            'response': fact,
        }
    )
    cards = entity_cards.EntityCardGenerator.decode(
        None, prediction, canonical_name='Group', fact=fact
    )
    assert len(cards) == 1
    assert cards[0].prompt == 'What element must a group have?'
    assert cards[0].response == fact


def test_create_entity_cards_routes_facts_and_creates_one_card_per_fact():
    router = _Router(True)
    fact_generator = _FactGenerator(
        [
            'A group has an identity element.',
            'A group operation is associative.',
        ]
    )
    card_generator = _CardGenerator()

    cards = asyncio.run(
        entity_cards.create_entity_cards(
            'hub-1',
            'Group',
            'A group has an identity element and an associative operation.',
            router=router,
            fact_generator=fact_generator,
            card_generator=card_generator,
        )
    )

    assert len(cards) == 2
    assert router.calls == [
        {
            'canonical_name': 'Group',
            'description': (
                'A group has an identity element and an associative operation.'
            ),
        }
    ]
    assert len(fact_generator.calls) == 1
    assert len(card_generator.calls) == 2
    assert all(card.hub_uuid == 'hub-1' for card in cards)
    assert all(card.hub_kind == 'entity' for card in cards)
    assert cards[0].prompt == 'What property does it have?'
    assert cards[0].response == 'A group has an identity element.'


def test_create_entity_cards_skips_fact_generation_when_router_rejects():
    router = _Router(False)
    fact_generator = _FactGenerator(['should not be used'])
    card_generator = _CardGenerator()

    cards = asyncio.run(
        entity_cards.create_entity_cards(
            'hub-1',
            'Group',
            'A vague description.',
            router=router,
            fact_generator=fact_generator,
            card_generator=card_generator,
        )
    )

    assert cards == []
    assert fact_generator.calls == []
    assert card_generator.calls == []