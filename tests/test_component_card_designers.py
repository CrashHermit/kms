import asyncio

import pytest

from kms.core import models
from kms.postprocessing.learning import (
    cards,
    entity_cards,
    procedure_cards,
    triplet_cards,
)


class _Suitability:
    def __init__(self, eligible=True):
        self.eligible = eligible

    async def aforward(self, **kwargs):
        return self.eligible


class _Generator:
    async def aforward(self, **kwargs):
        return cards.CardDraft(
            content_key='overview', prompt='What?', response='Answer'
        )


class _ProcedureGenerator:
    async def aforward(self, **kwargs):
        return [
            cards.CardDraft(
                content_key='step:1', prompt='What?', response='Answer'
            )
        ]


class _Verifier:
    def __init__(self, supported=True):
        self.supported = supported

    async def aforward(self, **kwargs):
        return cards.CardVerification(supported=self.supported)


def _input(kind, context=None):
    return cards.CardWorkerInput(
        hub_context=cards.CardHubContextInput(local_uuid='hub', source='book'),
        target=cards.CardTargetInput(
            uuid='target',
            kind=kind,
            source='book',
            content='Subject --does not cause--> Object',
            context=context or {},
        ),
    )


def test_entity_cards_are_owned_by_entity_target():
    designer = entity_cards.EntityCardDesigner(
        _Suitability(), _Generator(), _Verifier()
    )
    result = asyncio.run(
        designer.create_cards(_input(models.CardTargetKind.ENTITY))
    )
    assert result.cards[0].target_kind == models.CardTargetKind.ENTITY
    assert result.cards[0].target_uuid == 'target'


def test_entity_rejection_is_ordinary_result():
    designer = entity_cards.EntityCardDesigner(
        _Suitability(False), _Generator(), _Verifier()
    )
    result = asyncio.run(
        designer.create_cards(_input(models.CardTargetKind.ENTITY))
    )
    assert not result.eligible
    assert not result.cards


def test_triplet_cards_preserve_target_identity_for_negated_relation():
    designer = triplet_cards.TripletCardDesigner(
        _Suitability(), _Generator(), _Verifier()
    )
    result = asyncio.run(
        designer.create_cards(_input(models.CardTargetKind.TRIPLET))
    )
    assert result.cards[0].target_kind == models.CardTargetKind.TRIPLET
    assert result.cards[0].target_uuid == 'target'


def test_triplet_designer_rejects_wrong_target_kind():
    designer = triplet_cards.TripletCardDesigner(
        _Suitability(), _Generator(), _Verifier()
    )
    with pytest.raises(ValueError, match='non-triplet'):
        asyncio.run(designer.create_cards(_input(models.CardTargetKind.EVENT)))


def test_generated_procedure_is_rejected_without_model_calls():
    designer = procedure_cards.ProcedureCardDesigner(
        _Suitability(), _ProcedureGenerator(), _Verifier()
    )
    result = asyncio.run(
        designer.create_cards(
            _input(models.CardTargetKind.PROCEDURE, {'kind': 'generated'})
        )
    )
    assert not result.eligible
    assert result.rejection_reason == 'generated procedure'


def test_source_procedure_creates_stable_variant():
    designer = procedure_cards.ProcedureCardDesigner(
        _Suitability(), _ProcedureGenerator(), _Verifier()
    )
    result = asyncio.run(
        designer.create_cards(_input(models.CardTargetKind.PROCEDURE))
    )
    assert result.cards[0].target_kind == models.CardTargetKind.PROCEDURE
    assert result.cards[0].uuid
