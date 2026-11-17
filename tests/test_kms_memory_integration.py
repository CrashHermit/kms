"""Real Neo4j coverage for durable owner-scoped memory reviews."""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fsrs import Card, Rating, Scheduler
from neo4j.exceptions import ConstraintError, ResultNotSingleError

from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.learning.memory.models import (
    MemoryReviewState,
    memory_uuid,
    review_uuid,
)
from kms.learning.memory.node import MemoryReviewNode
from kms.learning.memory.repository import MemoryRepository
from kms.learning.references import (
    ArgumentTarget,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.task.models import ReviewRating

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


def _assessment_payload(rating: str) -> str:
    return json.dumps(
        {
            'grade': {
                'correctness': 'correct',
                'interpretation': {
                    'reasoning_verdict': 'sound',
                    'explanation': 'The response supports the answer.',
                    'demonstrated_understandings': ['Understands the target.'],
                    'gaps': [],
                    'possible_misconceptions': [],
                    'unresolved_questions': [],
                },
                'review_rating': rating,
            }
        },
        separators=(',', ':'),
    )


def _step_rows(
    prefix: str,
    proposition_uuid: str,
    argument_uuid: str,
    solution_uuid: str,
    foreign_proposition_uuid: str,
    now: datetime,
) -> list[dict[str, object]]:
    attempts = [
        ('prop-first', 'proposition', proposition_uuid, now, 'good'),
        ('prop-concurrent-a', 'proposition', proposition_uuid, now, 'hard'),
        ('prop-concurrent-b', 'proposition', proposition_uuid, now, 'easy'),
        (
            'prop-stale',
            'proposition',
            proposition_uuid,
            now - timedelta(days=1),
            'again',
        ),
        ('argument', 'argument', argument_uuid, now, 'good'),
        ('solution', 'solution', solution_uuid, now, 'again'),
        (
            'foreign-target',
            'proposition',
            foreign_proposition_uuid,
            now,
            'good',
        ),
    ]
    rows = []
    for suffix, kind, target_uuid, reviewed_at, rating in attempts:
        task_uuid = f'{prefix}-task-{suffix}'
        response_uuid = f'{prefix}-response-{suffix}'
        rows.append(
            {
                'session_uuid': f'{prefix}-session-{suffix}',
                'task_uuid': task_uuid,
                'response_uuid': response_uuid,
                'assessment_uuid': f'{prefix}-assessment-{suffix}',
                'target_kind': kind,
                'target_uuid': target_uuid,
                'solution_uuid': f'{prefix}-owned-solution-{suffix}',
                'reviewed_at': reviewed_at,
                'rating': rating,
                'task_payload_json': json.dumps(
                    {
                        'kind': 'task',
                        'understanding': f'Understanding {suffix}',
                        'prompt': {'parts': []},
                    },
                    separators=(',', ':'),
                ),
                'solution_payload_json': json.dumps(
                    {
                        'material': {'parts': []},
                        'assessment_criteria': ['Provides a supported answer.'],
                    },
                    separators=(',', ':'),
                ),
                'assessment_payload_json': _assessment_payload(rating),
            }
        )
    rows.append(
        {
            'session_uuid': f'{prefix}-session-solution-owner',
            'task_uuid': f'{prefix}-task-solution-owner',
            'response_uuid': None,
            'assessment_uuid': None,
            'target_kind': 'proposition',
            'target_uuid': proposition_uuid,
            'solution_uuid': solution_uuid,
            'reviewed_at': now,
            'rating': 'good',
            'task_payload_json': json.dumps(
                {
                    'kind': 'task',
                    'understanding': 'Original solution problem',
                    'prompt': {'parts': []},
                },
                separators=(',', ':'),
            ),
            'solution_payload_json': json.dumps(
                {
                    'material': {'parts': []},
                    'assessment_criteria': ['The original solution.'],
                },
                separators=(',', ':'),
            ),
            'assessment_payload_json': None,
        }
    )
    return [rows[-1], *rows[:-1]]


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_memory_reviews_are_owned_atomic_and_ordered() -> None:
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        repository = MemoryRepository(database.session)
        node = MemoryReviewNode(Scheduler(enable_fuzzing=False), repository)
        prefix = f'memory-it-{uuid4().hex}'
        owner_uuid = f'{prefix}-owner'
        empty_owner_uuid = f'{prefix}-empty-owner'
        foreign_owner_uuid = f'{prefix}-foreign-owner'
        proposition_uuid = f'{prefix}-shared-target'
        premise_uuid = f'{prefix}-premise'
        conclusion_uuid = f'{prefix}-conclusion'
        argument_uuid = proposition_uuid
        foreign_proposition_uuid = f'{prefix}-foreign-proposition'
        overdue_proposition_uuid = f'{prefix}-overdue-proposition'
        overdue_memory_uuid = memory_uuid(
            owner_uuid, PropositionTarget(uuid=overdue_proposition_uuid)
        )
        premise_memory_uuid = memory_uuid(
            owner_uuid, PropositionTarget(uuid=premise_uuid)
        )
        solution_uuid = f'{prefix}-solution-target'
        foreign_memory_uuid = memory_uuid(
            foreign_owner_uuid, PropositionTarget(uuid=foreign_proposition_uuid)
        )
        now = datetime.now(UTC).replace(microsecond=0)
        rows = _step_rows(
            prefix,
            proposition_uuid,
            argument_uuid,
            solution_uuid,
            foreign_proposition_uuid,
            now,
        )
        fixture_uuids = {
            owner_uuid,
            empty_owner_uuid,
            foreign_owner_uuid,
            proposition_uuid,
            premise_uuid,
            conclusion_uuid,
            foreign_proposition_uuid,
            overdue_proposition_uuid,
            *[row['session_uuid'] for row in rows],
            *[row['task_uuid'] for row in rows],
            *[row['solution_uuid'] for row in rows],
            *[row['response_uuid'] for row in rows if row['response_uuid']],
            *[row['assessment_uuid'] for row in rows if row['assessment_uuid']],
            memory_uuid(owner_uuid, PropositionTarget(uuid=proposition_uuid)),
            memory_uuid(owner_uuid, ArgumentTarget(uuid=argument_uuid)),
            foreign_memory_uuid,
            overdue_memory_uuid,
            premise_memory_uuid,
            *[
                review_uuid(owner_uuid, row['response_uuid'])
                for row in rows
                if row['response_uuid']
            ],
        }
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    """
                    CREATE (owner:User {uuid: $owner_uuid})
                    CREATE (:User {uuid: $empty_owner_uuid})
                    CREATE (foreign:User {uuid: $foreign_owner_uuid})
                    CREATE (proposition:Proposition {
                        uuid: $proposition_uuid, name: 'Shared identity',
                        statement: 'The proposition target is owned.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (premise:Proposition {
                        uuid: $premise_uuid, name: 'Premise',
                        statement: 'A premise of the argument.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (conclusion:Proposition {
                        uuid: $conclusion_uuid, name: 'Conclusion',
                        statement: 'The argument conclusion.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (foreign_proposition:Proposition {
                        uuid: $foreign_proposition_uuid, name: 'Foreign',
                        statement: 'Owned by another user.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (overdue_proposition:Proposition {
                        uuid: $overdue_proposition_uuid, name: 'Overdue',
                        statement: 'An overdue target.',
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (argument:Argument {
                        uuid: $argument_uuid, created_at: datetime(),
                        updated_at: datetime()
                    })
                    CREATE (owner)-[:OWNS_PROPOSITION]->(proposition)
                    CREATE (owner)-[:OWNS_PROPOSITION]->(premise)
                    CREATE (owner)-[:OWNS_PROPOSITION]->(conclusion)
                    CREATE (owner)-[:OWNS_ARGUMENT]->(argument)
                    CREATE (owner)-[:OWNS_PROPOSITION]->(overdue_proposition)
                    CREATE (foreign)-[:OWNS_PROPOSITION]->(foreign_proposition)
                    CREATE (argument)-[:HAS_PREMISE]->(premise)
                    CREATE (argument)-[:CONCLUDES]->(conclusion)
                    """,
                    owner_uuid=owner_uuid,
                    empty_owner_uuid=empty_owner_uuid,
                    foreign_owner_uuid=foreign_owner_uuid,
                    proposition_uuid=proposition_uuid,
                    premise_uuid=premise_uuid,
                    conclusion_uuid=conclusion_uuid,
                    foreign_proposition_uuid=foreign_proposition_uuid,
                    overdue_proposition_uuid=overdue_proposition_uuid,
                    argument_uuid=argument_uuid,
                )
                await result.consume()
                base = rows[0]
                result = await session.run(
                    """
                    MATCH (owner:User {uuid: $owner_uuid})
                    MATCH (owner)-[:OWNS_PROPOSITION]->(target:Proposition {uuid: $target_uuid})
                    CREATE (learning_session:LearningSession {
                        uuid: $session_uuid, status: 'active',
                        goal: 'Original solution task', targets: [],
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (owner)-[:OWNS_SESSION]->(learning_session)
                    CREATE (task:Task {
                        uuid: $task_uuid, payload_json: $task_payload_json,
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (learning_session)-[:HAS_STEP]->(task)
                    CREATE (task)-[:ASSESSES]->(target)
                    CREATE (solution:Solution {
                        uuid: $solution_uuid, payload_json: $solution_payload_json,
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (task)-[:HAS_SOLUTION]->(solution)
                    RETURN task.uuid AS uuid
                    """,
                    owner_uuid=owner_uuid,
                    target_uuid=proposition_uuid,
                    session_uuid=base['session_uuid'],
                    task_uuid=base['task_uuid'],
                    task_payload_json=base['task_payload_json'],
                    solution_uuid=base['solution_uuid'],
                    solution_payload_json=base['solution_payload_json'],
                )
                await result.consume()
                result = await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (owner:User {uuid: $owner_uuid})
                    CREATE (learning_session:LearningSession {
                        uuid: row.session_uuid, status: 'active',
                        goal: 'Review a single target', targets: [],
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (owner)-[:OWNS_SESSION]->(learning_session)
                    CREATE (task:Task {
                        uuid: row.task_uuid,
                        payload_json: row.task_payload_json,
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (learning_session)-[:HAS_STEP]->(task)
                    CREATE (solution:Solution {
                        uuid: row.solution_uuid,
                        payload_json: row.solution_payload_json,
                        created_at: datetime(), updated_at: datetime()
                    })
                    CREATE (task)-[:HAS_SOLUTION]->(solution)
                    WITH owner, row, learning_session, task
                    CALL (owner, row, learning_session, task) {
                        WITH owner, row, learning_session, task
                        WHERE row.target_kind = 'proposition'
                        MATCH (owner)-[:OWNS_PROPOSITION]->(target:Proposition {uuid: row.target_uuid})
                        CREATE (task)-[:ASSESSES]->(target)
                        RETURN target AS matched_target
                        UNION
                        WITH owner, row, learning_session, task
                        WHERE row.target_kind = 'argument'
                        MATCH (owner)-[:OWNS_ARGUMENT]->(target:Argument {uuid: row.target_uuid})
                        CREATE (task)-[:ASSESSES]->(target)
                        RETURN target AS matched_target
                        UNION
                        WITH owner, row, learning_session, task
                        WHERE row.target_kind = 'solution'
                        MATCH (owner)-[:OWNS_SESSION]->(:LearningSession)-[:HAS_STEP]->(:Task)-[:HAS_SOLUTION]->(target:Solution {uuid: row.target_uuid})
                        CREATE (task)-[:ASSESSES]->(target)
                        RETURN target AS matched_target
                    }
                    WITH row, learning_session, task
                    WHERE row.response_uuid IS NOT NULL
                    CREATE (response:TaskResponse {
                        uuid: row.response_uuid, payload_json: '{}',
                        created_at: row.reviewed_at, updated_at: row.reviewed_at
                    })
                    CREATE (task)-[:HAS_RESPONSE]->(response)
                    CREATE (assessment:TaskAssessment {
                        uuid: row.assessment_uuid,
                        payload_json: row.assessment_payload_json,
                        created_at: row.reviewed_at + duration({minutes: 30}),
                        updated_at: row.reviewed_at + duration({minutes: 30})
                    })
                    CREATE (response)-[:HAS_ASSESSMENT]->(assessment)
                    RETURN count(*) AS count
                    """,
                    owner_uuid=owner_uuid,
                    rows=rows[1:],
                )
                await result.consume()

            first_state: MemoryReviewState = {
                'user_uuid': owner_uuid,
                'session_uuid': rows[1]['session_uuid'],
                'assessment_uuid': rows[1]['assessment_uuid'],
            }
            first = (await node.run(first_state))['memory']
            assert first.target == PropositionTarget(uuid=proposition_uuid)
            assert first.review_count == 1
            assert first.due_at > now

            race_states: list[MemoryReviewState] = [
                {
                    'user_uuid': owner_uuid,
                    'session_uuid': row['session_uuid'],
                    'assessment_uuid': row['assessment_uuid'],
                }
                for row in rows[2:4]
            ]
            await asyncio.gather(*(node.run(state) for state in race_states))
            history = await repository.read_memory(
                owner_uuid, PropositionTarget(uuid=proposition_uuid)
            )
            assert history.memory.review_count == 3
            assert [review.sequence for review in history.reviews] == [1, 2, 3]
            assert {review.assessment_uuid for review in history.reviews} == {
                rows[index]['assessment_uuid'] for index in range(1, 4)
            }
            assert {review.reviewed_at for review in history.reviews} == {now}
            replay = Card(card_id=UUID(history.memory.uuid).int, due=now)
            scheduler = Scheduler(enable_fuzzing=False)
            ratings = {
                ReviewRating.AGAIN: Rating.Again,
                ReviewRating.HARD: Rating.Hard,
                ReviewRating.GOOD: Rating.Good,
                ReviewRating.EASY: Rating.Easy,
            }
            for review in history.reviews:
                replay, _ = scheduler.review_card(
                    replay,
                    ratings[review.rating],
                    review_datetime=review.reviewed_at,
                )
            assert replay.to_json() == history.memory.card_json
            assert replay.due == history.memory.due_at
            assert {
                review.assessment_uuid: review.rating
                for review in history.reviews
            } == {
                rows[1]['assessment_uuid']: ReviewRating.GOOD,
                rows[2]['assessment_uuid']: ReviewRating.HARD,
                rows[3]['assessment_uuid']: ReviewRating.EASY,
            }
            previous_memory = history.memory
            previous_reviews = history.reviews
            with pytest.raises(ConstraintError):
                await node.run(first_state)
            after_duplicate = await repository.read_memory(
                owner_uuid, PropositionTarget(uuid=proposition_uuid)
            )
            assert after_duplicate.memory == previous_memory
            assert after_duplicate.reviews == previous_reviews

            stale_state: MemoryReviewState = {
                'user_uuid': owner_uuid,
                'session_uuid': rows[4]['session_uuid'],
                'assessment_uuid': rows[4]['assessment_uuid'],
            }
            with pytest.raises(
                ValueError,
                match='Review precedes the latest recorded attempt',
            ):
                await node.run(stale_state)
            assert (
                await repository.read_memory(
                    owner_uuid, PropositionTarget(uuid=proposition_uuid)
                )
                == after_duplicate
            )

            argument_state: MemoryReviewState = {
                'user_uuid': owner_uuid,
                'session_uuid': rows[5]['session_uuid'],
                'assessment_uuid': rows[5]['assessment_uuid'],
            }
            solution_state: MemoryReviewState = {
                'user_uuid': owner_uuid,
                'session_uuid': rows[6]['session_uuid'],
                'assessment_uuid': rows[6]['assessment_uuid'],
            }
            argument_memory = (await node.run(argument_state))['memory']
            solution_memory = (await node.run(solution_state))['memory']
            assert argument_memory.uuid == memory_uuid(
                owner_uuid, ArgumentTarget(uuid=argument_uuid)
            )
            assert argument_memory.uuid != first.uuid
            assert solution_memory.target == SolutionTarget(uuid=solution_uuid)
            assert solution_memory.review_count == 1

            proposition_context = await repository.read_target(
                owner_uuid, PropositionTarget(uuid=proposition_uuid)
            )
            assert proposition_context.understanding == (
                'The proposition target is owned.'
            )
            assert proposition_context.knowledge.matched[0].uuid == (
                proposition_uuid
            )
            target_context = await repository.read_target(
                owner_uuid, ArgumentTarget(uuid=argument_uuid)
            )
            assert target_context.understanding == 'The argument conclusion.'
            assert target_context.knowledge.matched == []
            assert {
                item.uuid for item in target_context.knowledge.propositions
            } == {
                premise_uuid,
                conclusion_uuid,
            }
            assert target_context.knowledge.arguments[0].uuid == argument_uuid
            solution_context = await repository.read_target(
                owner_uuid, SolutionTarget(uuid=solution_uuid)
            )
            assert solution_context.understanding == 'Original solution problem'
            assert solution_context.solution_task is not None
            assert solution_context.solution_task.solution.uuid == solution_uuid

            foreign_state: MemoryReviewState = {
                'user_uuid': owner_uuid,
                'session_uuid': rows[7]['session_uuid'],
                'assessment_uuid': rows[7]['assessment_uuid'],
            }
            with pytest.raises(ResultNotSingleError):
                await node.run(foreign_state)
            with pytest.raises(ResultNotSingleError):
                await repository.list_due(
                    f'{prefix}-missing-owner', now + timedelta(days=1)
                )
            assert (
                await repository.list_due(
                    empty_owner_uuid, now + timedelta(days=1)
                )
                == []
            )

            due_at = now + timedelta(days=1)
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (owner:User {uuid: $owner_uuid})-[:OWNS_PROPOSITION]->(target:Proposition {uuid: $target_uuid})
                    CREATE (owner)-[:HAS_MEMORY {
                        created_at: $now, updated_at: $now
                    }]->(memory:LearningMemory {
                        uuid: $memory_uuid, card_json: '{}', due_at: $due_at,
                        review_count: 1, created_at: $now,
                        updated_at: $now, invalidated_at: null
                    })
                    CREATE (memory)-[:FOR_TARGET {
                        created_at: $now, updated_at: $now
                    }]->(target)
                    RETURN memory.uuid AS uuid
                    """,
                    owner_uuid=foreign_owner_uuid,
                    target_uuid=foreign_proposition_uuid,
                    memory_uuid=foreign_memory_uuid,
                    now=now,
                    due_at=now,
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (owner:User {uuid: $owner_uuid})-[:OWNS_PROPOSITION]->(target:Proposition {uuid: $target_uuid})
                    CREATE (owner)-[:HAS_MEMORY {
                        created_at: $now, updated_at: $now
                    }]->(memory:LearningMemory {
                        uuid: $memory_uuid, card_json: '{}',
                        due_at: $due_at, review_count: 1,
                        created_at: $now, updated_at: $now,
                        invalidated_at: null
                    })
                    CREATE (memory)-[:FOR_TARGET {
                        created_at: $now, updated_at: $now
                    }]->(target)
                    RETURN memory.uuid AS uuid
                    """,
                    owner_uuid=owner_uuid,
                    target_uuid=overdue_proposition_uuid,
                    memory_uuid=overdue_memory_uuid,
                    now=now,
                    due_at=now - timedelta(days=1),
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (owner:User {uuid: $owner_uuid})-[:OWNS_PROPOSITION]->(target:Proposition {uuid: $target_uuid})
                    CREATE (owner)-[:HAS_MEMORY {
                        created_at: $now, updated_at: $now
                    }]->(memory:LearningMemory {
                        uuid: $memory_uuid, card_json: '{}',
                        due_at: $due_at, review_count: 1,
                        created_at: $now, updated_at: $now,
                        invalidated_at: null
                    })
                    CREATE (memory)-[:FOR_TARGET {
                        created_at: $now, updated_at: $now
                    }]->(target)
                    RETURN memory.uuid AS uuid
                    """,
                    owner_uuid=owner_uuid,
                    target_uuid=premise_uuid,
                    memory_uuid=premise_memory_uuid,
                    now=now,
                    due_at=now,
                )
                await result.consume()

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (owner:User {uuid: $owner_uuid})-[:HAS_MEMORY]->(memory:LearningMemory)
                    MATCH (memory)-[:FOR_TARGET]->(target)
                    SET memory.due_at = CASE
                        WHEN target.uuid = $overdue_target_uuid
                            THEN $due_at - duration({days: 1})
                        WHEN target:Proposition THEN $due_at
                        WHEN target:Argument THEN $due_at
                        ELSE $due_at + duration({days: 1})
                    END
                    RETURN count(memory) AS count
                    """,
                    owner_uuid=owner_uuid,
                    overdue_target_uuid=overdue_proposition_uuid,
                    due_at=due_at,
                )
                await result.consume()
            due = await repository.list_due(owner_uuid, due_at)
            assert [item.target.kind for item in due] == [
                'proposition',
                'argument',
                'proposition',
                'proposition',
            ]
            assert [item.target.uuid for item in due] == [
                overdue_proposition_uuid,
                argument_uuid,
                *sorted([premise_uuid, proposition_uuid]),
            ]
            assert [item.due_at for item in due] == [
                due_at - timedelta(days=1),
                due_at,
                due_at,
                due_at,
            ]
            assert len(due) == 4
            assert foreign_memory_uuid not in {item.uuid for item in due}
            assert first.uuid != solution_memory.uuid

        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (node) WHERE node.uuid IN $uuids DETACH DELETE node',
                    uuids=list(fixture_uuids),
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
