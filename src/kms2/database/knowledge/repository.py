"""User-owned knowledge reads and atomic admission writes."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from neo4j import AsyncSession, AsyncTransaction, Record
from neo4j.time import DateTime

from kms2.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeAssertionWrite,
)
from kms2.core.model.knowledge.proposition import (
    KnowledgeAssertion,
    KnowledgeAssertionRecord,
    Proposition,
    PropositionRecord,
    ReasoningStep,
)
from kms2.core.model.knowledge.reasoning import ReasoningStepRecord
from kms2.core.model.knowledge.retrieval import PropositionSearchCandidate
from kms2.database.knowledge.queries.admission import (
    CREATE_KNOWLEDGE_ASSERTION,
    CREATE_KNOWLEDGE_PROPOSITION,
    CREATE_KNOWLEDGE_REASONING_STEP,
    CREATE_KNOWLEDGE_REASONING_STEP_WITHOUT_PREMISES,
    INCREMENT_KNOWLEDGE_REVISION,
    LINK_EXISTING_KNOWLEDGE_REASONING_STEP,
    LINK_KNOWLEDGE_ASSERTION_REASONING,
    LINK_KNOWLEDGE_ASSERTION_SUPPORT,
    LINK_KNOWLEDGE_STEP_SUPPORT,
)
from kms2.database.knowledge.queries.retrieval import (
    FIND_EXACT_OWNED_PROPOSITION,
    FIND_OWNED_REASONING_STEP_CANDIDATES,
    READ_KNOWLEDGE_REVISION,
    READ_OWNED_PROPOSITION,
    READ_OWNED_REASONING_STEP,
    SEARCH_OWNED_PROPOSITIONS,
)


class KnowledgeRepository:
    """Read and search only propositions owned by the requested user."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def read_revision(self, user_uuid: str) -> int:
        """Read the user's admission revision, treating its initial value as zero."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_KNOWLEDGE_REVISION,
                user_uuid=user_uuid,
            )
            record = await result.single(strict=True)
        return int(record['revision'])

    async def search_propositions(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
    ) -> list[PropositionSearchCandidate]:
        """Return indexed cosine-ranked propositions owned by the user."""
        async with self._session_factory() as session:
            result = await session.run(
                SEARCH_OWNED_PROPOSITIONS,
                user_uuid=user_uuid,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
            )
            rows = await result.data()
        return [
            PropositionSearchCandidate(
                proposition=_parse_proposition(row['proposition']),
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def read_propositions(
        self,
        user_uuid: str,
        proposition_uuids: list[str],
    ) -> list[PropositionRecord]:
        """Read requested owned propositions and bounded direct provenance."""
        requested = list(dict.fromkeys(proposition_uuids))
        if not requested:
            return []
        async with self._session_factory() as session:
            records: list[PropositionRecord] = []
            for proposition_uuid in requested:
                result = await session.run(
                    READ_OWNED_PROPOSITION,
                    user_uuid=user_uuid,
                    proposition_uuid=proposition_uuid,
                )
                row = await result.single(strict=True)
                records.append(_parse_proposition_record(row))
        return records

    async def read_reasoning_steps(
        self,
        user_uuid: str,
        step_uuids: list[str],
    ) -> list[ReasoningStepRecord]:
        """Read each requested owned argument with only direct references."""
        requested = list(dict.fromkeys(step_uuids))
        if not requested:
            return []
        async with self._session_factory() as session:
            records: list[ReasoningStepRecord] = []
            for step_uuid in requested:
                result = await session.run(
                    READ_OWNED_REASONING_STEP,
                    user_uuid=user_uuid,
                    reasoning_step_uuid=step_uuid,
                )
                row = await result.single(strict=True)
                records.append(_parse_reasoning_record(row))
        return records

    async def find_exact_proposition(
        self,
        user_uuid: str,
        content: str,
    ) -> Proposition | None:
        """Find the oldest exact-content proposition in this user's library."""
        async with self._session_factory() as session:
            result = await session.run(
                FIND_EXACT_OWNED_PROPOSITION,
                user_uuid=user_uuid,
                content=content,
            )
            row = await result.single()
        return _parse_proposition(row['proposition']) if row else None

    async def find_reasoning_step(
        self,
        user_uuid: str,
        premise_uuids: list[str],
        conclusion_uuid: str,
        explanation: str,
        supporting_block_uuids: list[str],
    ) -> ReasoningStepRecord | None:
        """Find an owned exactly repeated joint-premise argument."""
        async with self._session_factory() as session:
            result = await session.run(
                FIND_OWNED_REASONING_STEP_CANDIDATES,
                user_uuid=user_uuid,
                conclusion_uuid=conclusion_uuid,
                explanation=explanation,
            )
            rows = await result.data()
        expected_premises = sorted(set(premise_uuids))
        expected_support = sorted(set(supporting_block_uuids))
        for row in rows:
            record = _parse_reasoning_record(row)
            if (
                record.premise_uuids == expected_premises
                and record.supporting_block_uuids == expected_support
            ):
                return record
        return None


async def write_knowledge_admission(
    transaction: AsyncTransaction,
    user_uuid: str,
    turn_uuid: str,
    admission: KnowledgeAdmission,
) -> None:
    """Write a prepared admission inside its containing teaching transaction.

    The caller has acquired the User and source locks, checked expected
    revisions, and created the owning teacher turn. This function performs no
    inference, session creation, or independent transaction.
    """
    for proposition in admission.new_propositions:
        row = await _single(
            transaction,
            CREATE_KNOWLEDGE_PROPOSITION,
            user_uuid=user_uuid,
            uuid=proposition.uuid,
            content=proposition.content,
            embedding=proposition.embedding,
            created_at=proposition.created_at,
        )
        if row['uuid'] != proposition.uuid:
            raise RuntimeError('Persisted proposition identity changed')

    for record in admission.new_reasoning_steps:
        step = record.step
        query = (
            CREATE_KNOWLEDGE_REASONING_STEP
            if record.premise_uuids
            else CREATE_KNOWLEDGE_REASONING_STEP_WITHOUT_PREMISES
        )
        row = await _single(
            transaction,
            query,
            user_uuid=user_uuid,
            uuid=step.uuid,
            explanation=step.explanation,
            created_at=step.created_at,
            conclusion_uuid=record.conclusion_uuid,
            premise_uuids=sorted(set(record.premise_uuids)),
        )
        if row['uuid'] != step.uuid:
            raise RuntimeError('Persisted reasoning-step identity changed')
        for block_uuid in sorted(set(record.supporting_block_uuids)):
            await _single(
                transaction,
                LINK_KNOWLEDGE_STEP_SUPPORT,
                user_uuid=user_uuid,
                step_uuid=step.uuid,
                block_uuid=block_uuid,
            )

    for step_uuid in sorted(set(admission.existing_reasoning_step_uuids)):
        await _single(
            transaction,
            LINK_EXISTING_KNOWLEDGE_REASONING_STEP,
            user_uuid=user_uuid,
            step_uuid=step_uuid,
        )

    for write in sorted(
        admission.assertions, key=lambda item: item.assertion.uuid
    ):
        await _write_assertion(transaction, user_uuid, turn_uuid, write)

    await _single(
        transaction,
        INCREMENT_KNOWLEDGE_REVISION,
        user_uuid=user_uuid,
    )


async def _write_assertion(
    transaction: AsyncTransaction,
    user_uuid: str,
    turn_uuid: str,
    write: KnowledgeAssertionWrite,
) -> None:
    row = await _single(
        transaction,
        CREATE_KNOWLEDGE_ASSERTION,
        user_uuid=user_uuid,
        turn_uuid=turn_uuid,
        uuid=write.assertion.uuid,
        created_at=write.assertion.created_at,
        proposition_uuid=write.proposition_uuid,
    )
    if row['uuid'] != write.assertion.uuid:
        raise RuntimeError('Persisted assertion identity changed')
    for block_uuid in sorted(set(write.supporting_block_uuids)):
        await _single(
            transaction,
            LINK_KNOWLEDGE_ASSERTION_SUPPORT,
            user_uuid=user_uuid,
            turn_uuid=turn_uuid,
            assertion_uuid=write.assertion.uuid,
            block_uuid=block_uuid,
        )
    for step_uuid in sorted(set(write.reasoning_step_uuids)):
        await _single(
            transaction,
            LINK_KNOWLEDGE_ASSERTION_REASONING,
            user_uuid=user_uuid,
            turn_uuid=turn_uuid,
            assertion_uuid=write.assertion.uuid,
            step_uuid=step_uuid,
        )


async def _single(
    transaction: AsyncTransaction,
    query: str,
    **parameters: object,
) -> Record:
    result = await transaction.run(query, **parameters)
    return await result.single(strict=True)


def _parse_proposition(record: dict[str, Any]) -> Proposition:
    return Proposition(
        uuid=record['uuid'],
        content=record['content'],
        embedding=record['embedding'],
        created_at=_native_datetime(record['created_at']),
    )


def _parse_assertion_record(record: dict[str, Any]) -> KnowledgeAssertionRecord:
    return KnowledgeAssertionRecord(
        assertion=KnowledgeAssertion(
            uuid=record['assertion']['uuid'],
            created_at=_native_datetime(record['assertion']['created_at']),
        ),
        supporting_block_uuids=sorted(record['supporting_block_uuids']),
        reasoning_step_uuids=sorted(record['reasoning_step_uuids']),
        proposition_uuid=record['proposition_uuid'],
        teaching_turn_uuid=record['teaching_turn_uuid'],
    )


def _parse_proposition_record(record: Record) -> PropositionRecord:
    assertions = [
        _parse_assertion_record(item)
        for item in record['assertions']
        if item['assertion'] is not None
    ]
    assertions.sort(key=lambda item: item.assertion.uuid)
    return PropositionRecord(
        proposition=_parse_proposition(record['proposition']),
        assertions=assertions,
        reasoning_step_uuids=sorted(record['reasoning_step_uuids']),
    )


def _parse_reasoning_record(record: Record) -> ReasoningStepRecord:
    step = record['step']
    return ReasoningStepRecord(
        step=ReasoningStep(
            uuid=step['uuid'],
            explanation=step['explanation'],
            created_at=_native_datetime(step['created_at']),
        ),
        premise_uuids=sorted(record['premise_uuids']),
        conclusion_uuid=record['conclusion_uuid'],
        supporting_block_uuids=sorted(record['supporting_block_uuids']),
    )


def _native_datetime(value: datetime | DateTime) -> datetime:
    return value.to_native() if isinstance(value, DateTime) else value
