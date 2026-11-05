import asyncio
import json
from datetime import UTC, datetime

import pytest

from kms.config.source_search import SourceSearchSettings
from kms.core.model.content import MultimodalContent
from kms.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationDraft,
    LearnerObservationEvidence,
    LearnerObservationKind,
    LearnerObservationRecord,
    LearnerObservationSearchCandidate,
    LearnerObservationWrite,
)
from kms.core.model.teaching.question import TeachingQuestionKind
from kms.core.model.teaching.session import (
    TeachingPhase,
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnRecord,
)
from kms.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms.database.teaching.observation_repository import (
    LearnerObservationRepository,
    write_learner_observations,
)
from kms.database.teaching.records import learner_observation_from_record
from kms.service.learner_observation_search import (
    LearnerObservationSearchService,
)
from kms.service.learner_observations import LearnerObservationService
from kms.service.teaching import TeachingService


class _Selection:
    def __init__(self, positions=None):
        self.positions = positions
        self.calls = []

    async def select(self, *, query, material_kind, candidates, content_images):
        assert content_images == []
        self.calls.append((query, material_kind, candidates))
        return (
            list(range(len(candidates)))
            if self.positions is None
            else self.positions
        )


class _EmptyTransaction:
    def __init__(self):
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))


class _SingleResult:
    def __init__(self):
        self.strict_values = []

    async def single(self, *, strict):
        self.strict_values.append(strict)
        return {'uuid': 'observation-1'}


class _WriteTransaction:
    def __init__(self):
        self.result = _SingleResult()
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return self.result


class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    async def embed(self, materials, *, content_images, input_type):
        self.calls.append((materials, content_images, input_type))
        return [[0.25, 0.75]]


class _FakeReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, *, content_images, top_n=None):
        self.calls.append((query, documents, content_images, top_n))
        return [
            {'index': 1, 'relevance_score': 0.95},
            {'index': 0, 'relevance_score': 0.61},
        ]


class _ObservationRepository:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    async def search(self, user_uuid, query_embedding, *, candidate_limit):
        self.calls.append((user_uuid, query_embedding, candidate_limit))
        return self.candidates


def _record(uuid, kind, topic, description, now):
    return LearnerObservationRecord(
        observation=LearnerObservation(
            uuid=uuid,
            kind=kind,
            topic=topic,
            description=description,
            embedding=[0.1, 0.2],
            created_at=now,
        ),
        evidence=[],
    )


def test_prepare_assigns_immutable_observations_to_visible_learner_turns():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    learner = TeachingTurn(
        uuid='learner-turn',
        role=TeachingRole.LEARNER,
        content='A nonzero vector cannot map to zero.',
        created_at=now,
    )
    snapshot = TeachingSessionSnapshot(
        session=TeachingSession(
            uuid='session-1',
            goal='Understand the kernel',
            status=TeachingStatus.AWAITING_TEACHER,
            phase=TeachingPhase.PROBING,
            revision=1,
            created_at=now,
            updated_at=now,
        ),
        turns=[
            TeachingTurnRecord(
                turn=learner,
                question=None,
                question_step_uuid=None,
                responds_to_question_uuid=None,
                presented_plan_uuid=None,
                accepted_plan_uuid=None,
            )
        ],
        current_assessment=None,
        current_plan=None,
        current_plan_acceptance=None,
        current_step_uuid=None,
        pending_question_uuid=None,
    )
    drafts = [
        LearnerObservationDraft(
            kind=LearnerObservationKind.MISCONCEPTION,
            topic='linear maps',
            description='Believes a nonzero vector cannot map to zero.',
            evidence_turn_positions=[0],
        ),
        LearnerObservationDraft(
            kind=LearnerObservationKind.DEMONSTRATED_UNDERSTANDING,
            topic='linear maps',
            description='Later identifies a nonzero kernel vector.',
            evidence_turn_positions=[0],
        ),
    ]
    service = LearnerObservationService(object())

    writes = service.prepare(
        drafts,
        snapshot,
        visible_turn_indexes=[0],
        embeddings=[[0.3, 0.4], [0.5, 0.6]],
    )

    assert [write.evidence_turn_uuids for write in writes] == [
        ['learner-turn'],
        ['learner-turn'],
    ]
    assert [write.observation.kind for write in writes] == [
        draft.kind for draft in drafts
    ]
    assert writes[0].observation.uuid != writes[1].observation.uuid
    assert all(write.observation.created_at.tzinfo == UTC for write in writes)
    record = LearnerObservationRecord(
        observation=writes[0].observation,
        evidence=[
            LearnerObservationEvidence(
                turn_uuid='learner-turn',
                session_uuid='session-1',
                content=learner.content,
                created_at=now,
                question=None,
            )
        ],
    )
    public = service.view(record)
    snapshot.turns[0].learner_observations.append(record)
    session_view = TeachingService._view(snapshot)

    assert 'embedding' not in public.model_dump()
    assert 'embedding' not in session_view.model_dump()


def test_record_hydration_projects_question_without_answer_key():
    now = datetime(2026, 10, 2, tzinfo=UTC)
    row = {
        'observation': {
            'uuid': 'observation-1',
            'kind': 'misconception',
            'topic': 'linear maps',
            'description': 'Misstates the kernel condition.',
            'embedding': [0.1, 0.2],
            'created_at': now,
        },
        'evidence': [
            {
                'turn_uuid': 'learner-turn',
                'session_uuid': 'session-1',
                'content': 'I selected option b.',
                'created_at': now,
                'question': {
                    'uuid': 'question-1',
                    'kind': TeachingQuestionKind.CHECK.value,
                    'prompt': 'Which statement is correct?',
                    'response_json': json.dumps(
                        {
                            'kind': 'multiple_choice',
                            'options': [
                                {
                                    'key': 'a',
                                    'content': 'A vector in the kernel',
                                },
                                {'key': 'b', 'content': 'Any nonzero vector'},
                            ],
                            'correct_option_key': 'a',
                        }
                    ),
                    'created_at': now,
                },
            }
        ],
    }

    record = learner_observation_from_record(row)
    safe = record.evidence[0].question.model_dump()

    assert safe['response']['options'][0]['content'] == (
        'A vector in the kernel'
    )
    assert 'correct_option_key' not in str(safe)
    assert 'embedding' in record.observation.model_dump()


def test_search_reranks_owner_scoped_candidates_without_synthesizing_scores():
    now = datetime(2026, 10, 3, tzinfo=UTC)
    records = [
        _record(
            'observation-1',
            LearnerObservationKind.GAP,
            'matrix',
            'Needs a definition.',
            now,
        ),
        _record(
            'observation-2',
            LearnerObservationKind.MISCONCEPTION,
            'kernel',
            'Misstates the kernel.',
            now,
        ),
    ]
    repository = _ObservationRepository(
        [
            LearnerObservationSearchCandidate(
                record=records[0], similarity=0.8
            ),
            LearnerObservationSearchCandidate(
                record=records[1], similarity=0.7
            ),
        ]
    )
    embedder = _FakeEmbedder()
    reranker = _FakeReranker()
    selection = _Selection([1, 0])
    service = LearnerObservationSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(candidate_limit=20, result_limit=2),
        selection=selection,
    )

    hits = asyncio.run(service.search('owner-1', 'kernel topic'))

    assert [hit.record.observation.uuid for hit in hits] == [
        'observation-1',
        'observation-2',
    ]
    assert [hit.similarity for hit in hits] == [0.8, 0.7]
    assert [hit.relevance_score for hit in hits] == [0.61, 0.95]
    assert embedder.calls == [
        ([MultimodalContent(text='kernel topic')], [], 'query')
    ]
    assert repository.calls == [('owner-1', [0.25, 0.75], 20)]
    assert [material.text for material in reranker.calls[0][1]] == [
        'gap\nmatrix\nNeeds a definition.',
        'misconception\nkernel\nMisstates the kernel.',
    ]
    assert reranker.calls[0][0].text == 'kernel topic'
    assert reranker.calls[0][2] == []
    assert [candidate.material.text for candidate in selection.calls[0][2]] == [
        'misconception\nkernel\nMisstates the kernel.',
        'gap\nmatrix\nNeeds a definition.',
    ]


def test_writing_zero_observations_does_not_query_transaction():
    transaction = _EmptyTransaction()

    asyncio.run(
        write_learner_observations(
            transaction,
            'owner-1',
            'teacher-turn',
            [],
        )
    )

    assert transaction.calls == []


def test_multiple_learner_turns_share_one_strict_write_result():
    now = datetime(2026, 10, 5, tzinfo=UTC)
    write = LearnerObservationWrite(
        observation=LearnerObservation(
            uuid='observation-1',
            kind=LearnerObservationKind.PARTIAL_UNDERSTANDING,
            topic='linear maps',
            description='Connects nullspace and zero output.',
            embedding=[0.1, 0.2],
            created_at=now,
        ),
        evidence_turn_uuids=['learner-1', 'learner-2'],
    )
    transaction = _WriteTransaction()

    asyncio.run(
        write_learner_observations(
            transaction,
            'owner-1',
            'teacher-1',
            [write],
        )
    )

    assert len(transaction.calls) == 1
    assert transaction.calls[0][1]['evidence_turn_uuids'] == [
        'learner-1',
        'learner-2',
    ]
    assert transaction.result.strict_values == [True]


def test_empty_observation_search_skips_reranking():
    repository = _ObservationRepository([])
    embedder = _FakeEmbedder()
    reranker = _FakeReranker()
    selection = _Selection()
    search = LearnerObservationSearchService(
        repository,
        embedder,
        reranker,
        SourceSearchSettings(),
        selection=selection,
    )

    hits = asyncio.run(search.search('owner-1', 'anything'))

    assert hits == []
    assert reranker.calls == []
    assert embedder.calls == [
        ([MultimodalContent(text='anything')], [], 'query')
    ]
    assert repository.calls == [('owner-1', [0.25, 0.75], 32)]
    assert selection.calls == []


class _ReadResult:
    def __init__(self, rows):
        self.rows = rows

    async def data(self):
        return self.rows


class _ReadSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return _ReadResult(self.rows)


def test_observation_read_uses_one_ordered_owner_scoped_batch():
    now = datetime(2026, 10, 6, tzinfo=UTC)
    records = [
        _record(
            'observation-1', LearnerObservationKind.GAP, 'vectors', 'A', now
        ),
        _record(
            'observation-2',
            LearnerObservationKind.MISCONCEPTION,
            'matrices',
            'B',
            now,
        ),
    ]
    session = _ReadSession(
        [
            {'position': position, 'record': record.model_dump()}
            for position, record in enumerate(records)
        ]
    )
    repository = LearnerObservationRepository(lambda: session)

    loaded = asyncio.run(
        repository.read('owner-1', ['observation-1', 'observation-2'])
    )

    assert [record.observation.uuid for record in loaded] == [
        'observation-1',
        'observation-2',
    ]
    assert len(session.calls) == 1
    assert session.calls[0][1] == {
        'user_uuid': 'owner-1',
        'observation_uuids': ['observation-1', 'observation-2'],
    }


def test_observation_read_rejects_missing_owned_rows_and_skips_empty_io():
    now = datetime(2026, 10, 7, tzinfo=UTC)
    record = _record(
        'observation-1', LearnerObservationKind.GAP, 'vectors', 'A', now
    )
    session = _ReadSession([{'position': 0, 'record': record.model_dump()}])
    repository = LearnerObservationRepository(lambda: session)

    with pytest.raises(
        ValueError,
        match='Requested learner observation is not owned or no longer available',
    ):
        asyncio.run(repository.read('owner-1', ['observation-1', 'foreign']))

    empty_session = _ReadSession([])
    empty_repository = LearnerObservationRepository(lambda: empty_session)
    assert asyncio.run(empty_repository.read('owner-1', [])) == []
    assert empty_session.calls == []
