import asyncio

from kms.postprocessing.learning import entity_cards as cards
from kms.postprocessing.learning.entity_cards import (
    EntityCardDraft,
    EntityCardFact,
    EntityCardVerification,
    EntityLearningAssessment,
    EntityLearningCandidate,
    EntityLearningCandidateInput,
    EntityLearningEvidenceInput,
    EntityLearningGlobalContext,
)


def test_orchestrator_persists_only_verified_cards(monkeypatch):
    candidate = EntityLearningCandidate(
        global_hub=EntityLearningGlobalContext(uuid='g', name='G', description='context'),
        local_hub=EntityLearningCandidateInput(index=1, local_uuid='l', name='N', description='local'),
        source='book',
        evidence=(EntityLearningEvidenceInput(index=1, uuid='e', name='N', description='fact'),),
    )
    assessment = EntityLearningAssessment(
        candidates=(candidate,), global_decisions=(True,), local_decisions=(True,),
        accepted=(candidate,), global_metrics={}, local_metrics={},
    )
    monkeypatch.setattr(
        cards, 'select_learnable_candidates',
        lambda **kwargs: _async_value(assessment),
    )
    persisted = []
    monkeypatch.setattr(cards.writer, 'persist_cards', lambda items, **kwargs: _async_collect(persisted, items))
    monkeypatch.setattr(cards.writer, 'persist_card_hub_edges', lambda items, **kwargs: _async_noop())
    monkeypatch.setattr(cards.writer, 'persist_card_evidence', lambda items, **kwargs: _async_noop())
    result = asyncio.run(cards.generate_entity_cards(
        session_factory=object(), global_gate=object(), evidence_gate=object(),
        router=_Router(), fact_generator=_Facts(), card_generator=_Generator(), verifier=_Verifier(),
    ))
    assert result['persisted'] == 1
    assert persisted[0][0].evidence_uuids == ('e',)


async def _async_value(value):
    return value


async def _async_collect(target, value):
    target.append(value)


async def _async_noop():
    return None


class _Router:
    async def aforward(self, **kwargs):
        return True


class _Facts:
    async def aforward(self, **kwargs):
        return [EntityCardFact(text='fact', evidence_indexes=[1])]


class _Generator:
    async def aforward(self, **kwargs):
        return [EntityCardDraft(prompt='What?', response='fact', fact_index=1)]


class _Verifier:
    async def aforward(self, **kwargs):
        return EntityCardVerification(supported=True)
