import asyncio

import pytest

from kms.postprocessing.learning import entity_cards


class _Router:
    def __init__(self, result):
        self.result = result

    async def aforward(self, **kwargs):
        return self.result


class _FactGenerator:
    def __init__(self, facts):
        self.facts = facts

    async def aforward(self, **kwargs):
        return self.facts


class _CardGenerator:
    async def aforward(self, **kwargs):
        return [
            entity_cards.EntityCardDraft(
                prompt='What is the fact?',
                response='Paraphrased fact.',
                fact_index=kwargs['fact_index'],
            )
        ]


class _Verifier:
    def __init__(self, supported=True):
        self.supported = supported
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return entity_cards.EntityCardVerification(supported=self.supported)


def candidate():
    return entity_cards.EntityCardCandidateInput(
        global_hub=entity_cards.EntityLearningGlobalContext(
            uuid='g', name='Global', description='Context'
        ),
        local_hub=entity_cards.EntityLearningCandidateInput(
            index=1,
            local_uuid='l',
            name='Group',
            description='A group has identity.',
        ),
        source='book',
        evidence=[
            entity_cards.EntityCardEvidenceInput(
                index=1,
                uuid='e1',
                name='Group',
                description='A group has identity.',
            )
        ],
    )


def test_fact_evidence_indexes_are_strict():
    with pytest.raises(ValueError):
        entity_cards.EntityCardFact(text='Fact.', evidence_indexes=[1, 1])


def test_create_entity_cards_generates_grounded_card():
    verifier = _Verifier()
    cards = asyncio.run(
        entity_cards.create_entity_cards(
            candidate(),
            router=_Router(True),
            fact_generator=_FactGenerator(
                [
                    entity_cards.EntityCardFact(
                        text='A group has identity.', evidence_indexes=[1]
                    )
                ]
            ),
            card_generator=_CardGenerator(),
            verifier=verifier,
        )
    )
    assert cards[0].uuid == entity_cards.learning.card_uuid(
        'l', content_key='A group has identity.'
    )
    assert cards[0].evidence_uuids == ('e1',)
    assert verifier.calls[0]['response'] == 'Paraphrased fact.'


def test_rejected_verification_creates_no_card():
    cards = asyncio.run(
        entity_cards.create_entity_cards(
            candidate(),
            router=_Router(True),
            fact_generator=_FactGenerator(
                [
                    entity_cards.EntityCardFact(
                        text='Fact.', evidence_indexes=[1]
                    )
                ]
            ),
            card_generator=_CardGenerator(),
            verifier=_Verifier(False),
        )
    )
    assert cards == []


def test_router_rejection_skips_fact_generation():
    facts = _FactGenerator([])
    cards = asyncio.run(
        entity_cards.create_entity_cards(
            candidate(),
            router=_Router(False),
            fact_generator=facts,
            card_generator=_CardGenerator(),
            verifier=_Verifier(),
        )
    )
    assert cards == []
    assert facts.facts == []


def test_verification_requires_entailment_flag():
    decision = entity_cards.EntityCardVerification(
        supported=True, evidence_entails_response=False
    )
    assert not all(
        (
            decision.supported,
            decision.evidence_entails_response,
            decision.single_fact,
            decision.prompt_is_recall,
            decision.no_answer_leakage,
            decision.complete_answer,
        )
    )
