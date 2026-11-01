import asyncio
import json
from datetime import UTC, datetime

from kms2.config.source_search import SourceSearchSettings
from kms2.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationDraft,
    LearnerObservationEvidence,
    LearnerObservationKind,
    LearnerObservationRecord,
    LearnerObservationSearchCandidate,
    LearnerObservationWrite,
)
from kms2.core.model.teaching.question import TeachingQuestionKind
from kms2.core.model.teaching.session import (
    TeachingPhase,
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnRecord,
)
from kms2.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms2.database.teaching.observation_repository import (
    write_learner_observations,
)
from kms2.database.teaching.records import learner_observation_from_record
from kms2.service.learner_observation_search import (
    LearnerObservationSearchService,
)
from kms2.service.learner_observations import LearnerObservationService
from kms2.service.teaching import TeachingService


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


class _EmbeddingClient:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return [[0.25, 0.75]]


class _RerankerClient:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append((query, documents, top_n))
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
    embedding_client = _EmbeddingClient()
    reranker_client = _RerankerClient()
    service = LearnerObservationSearchService(
        repository,
        embedding_client,
        reranker_client,
        SourceSearchSettings(candidate_limit=20, result_limit=2),
    )

    hits = asyncio.run(service.search('owner-1', 'kernel topic'))

    assert [hit.record.observation.uuid for hit in hits] == [
        'observation-2',
        'observation-1',
    ]
    assert [hit.relevance_score for hit in hits] == [0.95, 0.61]
    assert embedding_client.calls == [['kernel topic']]
    assert repository.calls == [('owner-1', [0.25, 0.75], 20)]
    assert reranker_client.calls[0][1] == [
        'gap\nmatrix\nNeeds a definition.',
        'misconception\nkernel\nMisstates the kernel.',
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
    embedding_client = _EmbeddingClient()
    reranker_client = _RerankerClient()
    search = LearnerObservationSearchService(
        repository,
        embedding_client,
        reranker_client,
        SourceSearchSettings(),
    )

    hits = asyncio.run(search.search('owner-1', 'anything'))

    assert hits == []
    assert reranker_client.calls == []
    assert embedding_client.calls == [['anything']]
    assert repository.calls == [('owner-1', [0.25, 0.75], 32)]
