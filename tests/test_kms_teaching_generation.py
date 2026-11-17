"""Behavioral checks for grounded teaching construction and persistence."""

import asyncio

import pytest

from kms.knowledge.models import KnowledgeContext, Proposition
from kms.learning.materials import ImageReference, MaterialDraft
from kms.learning.references import GroundingReferences, PropositionTarget
from kms.learning.session.models import SessionContext
from kms.learning.teaching.generation.models import (
    TeachingDraft,
    TeachingGenerationRequest,
    TeachingState,
)
from kms.learning.teaching.generation.node import TeachingGenerationNode
from kms.learning.teaching.models import Teaching
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.models.source import Source
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import SourceUnitMaterial, SourceUnitWindow


class _Module:
    def __init__(self, events: list[str], draft: TeachingDraft) -> None:
        self.events = events
        self.draft = draft

    async def acall(
        self, *, request: TeachingGenerationRequest
    ) -> TeachingDraft:
        self.events.append('inference')
        return self.draft


class _Repository:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.persisted: Teaching | None = None
        self.fail = fail

    async def read_session(self, user_uuid: str, session_uuid: str) -> object:
        self.events.append('read')
        return object()

    async def append_teaching(
        self, user_uuid: str, session_uuid: str, teaching: Teaching
    ) -> Teaching:
        self.events.append('persist')
        if self.fail:
            raise RuntimeError('write failed')
        self.persisted = teaching.model_copy(
            update={'uuid': 'persisted-teaching'}
        )
        return self.persisted


def _request(asset: ImageAsset) -> TeachingGenerationRequest:
    proposition = Proposition(
        uuid='proposition',
        name='Function value',
        statement='The filled point determines the function value.',
    )
    evidence = SourceUnitWindow(
        source=Source(uuid='source', key='graph'),
        before=[],
        target=SourceUnitMaterial(
            uuid='unit',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(text='Filled point is at 5: '),
                    ImagePart(asset=asset),
                )
            ),
        ),
        after=[],
    )
    return TeachingGenerationRequest(
        session=SessionContext(
            goal='Read the function value from a graph.',
            targets=['Identify the filled point.'],
            steps=[],
            responses=[],
            assessments=[],
        ),
        understanding='Read the filled point as the function value.',
        objective='Explain why the filled point, not the open point, gives f(2).',
        target=PropositionTarget(uuid=proposition.uuid),
        knowledge=KnowledgeContext(
            matched=[proposition], propositions=[proposition], arguments=[]
        ),
        evidence=[evidence],
        assessment_uuids=[],
    )


def _draft(image_position: int = 0) -> TeachingDraft:
    return TeachingDraft(
        material=MaterialDraft(
            parts=(
                TextPart(text='At x=2, the filled point gives f(2). '),
                ImageReference(image_position=image_position),
                ImageReference(image_position=image_position),
            )
        ),
        grounding=GroundingReferences(
            proposition_uuids=['proposition'], unit_uuids=['unit']
        ),
    )


def _state(request: TeachingGenerationRequest) -> TeachingState:
    return {'user_uuid': 'user', 'session_uuid': 'session', 'request': request}


def test_teaching_node_resolves_ordered_images_and_returns_persisted_step() -> (
    None
):
    asset = ImageAsset(('evidence',), b'evidence-image')
    events: list[str] = []
    repository = _Repository(events)
    request = _request(asset)

    result = asyncio.run(
        TeachingGenerationNode(_Module(events, _draft()), repository).run(
            _state(request)
        )
    )

    assert events == ['read', 'inference', 'persist']
    teaching = result['teaching']
    assert teaching is repository.persisted
    assert teaching.uuid == 'persisted-teaching'
    assert teaching.understanding == request.understanding
    assert teaching.objective == request.objective
    assert teaching.target == request.target
    assert teaching.assessment_uuids == request.assessment_uuids
    assert teaching.grounding == GroundingReferences(
        proposition_uuids=['proposition'], unit_uuids=['unit']
    )
    assert [part.type for part in teaching.material.parts] == [
        'text',
        'image',
        'image',
    ]
    assert teaching.material.images == (asset, asset)


def test_teaching_node_does_not_infer_for_foreign_session() -> None:
    class ForeignRepository(_Repository):
        async def read_session(
            self, user_uuid: str, session_uuid: str
        ) -> object:
            self.events.append('read')
            raise LookupError('foreign session')

    events: list[str] = []
    node = TeachingGenerationNode(
        _Module(events, _draft()), ForeignRepository(events)
    )

    with pytest.raises(LookupError, match='foreign session'):
        asyncio.run(
            node.run(_state(_request(ImageAsset(('evidence',), b'image'))))
        )

    assert events == ['read']


def test_teaching_node_propagates_persistence_failure() -> None:
    events: list[str] = []
    repository = _Repository(events, fail=True)
    node = TeachingGenerationNode(_Module(events, _draft()), repository)

    with pytest.raises(RuntimeError, match='write failed'):
        asyncio.run(
            node.run(_state(_request(ImageAsset(('evidence',), b'image'))))
        )

    assert events == ['read', 'inference', 'persist']
    assert repository.persisted is None
