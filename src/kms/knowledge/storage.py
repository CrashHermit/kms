"""Persistence and retrieval for owner-scoped propositions and arguments."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from neo4j import AsyncSession, AsyncTransaction

from kms.core.models import Edge
from kms.knowledge.models import Argument, KnowledgeContext, Proposition
from kms.knowledge.queries import (
    ATTACH_ARGUMENT_PROVENANCE,
    ATTACH_SOURCE_PROVENANCE,
    LOCK_KNOWLEDGE_OWNER,
    READ_CONTEXT,
    READ_MATCHED_PROPOSITIONS,
    READ_OWNED_ARGUMENTS,
    REQUIRE_OWNED_SOURCE_UNIT,
    SAVE_ARGUMENTS,
    SAVE_PROPOSITIONS,
    SEARCH_PROPOSITIONS,
)
from kms.source.storage.records import native_datetime


class KnowledgeRepository:
    """Persist and query knowledge graph objects within user ownership."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def serialized(
        self, user_uuid: str
    ) -> AsyncIterator[AsyncTransaction]:
        """Hold the owner's write lock in an explicit transaction."""
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                result = await transaction.run(
                    LOCK_KNOWLEDGE_OWNER, user_uuid=user_uuid
                )
                if await result.single() is None:
                    raise PermissionError('Knowledge owner does not exist')
                try:
                    yield transaction
                except BaseException:
                    await transaction.rollback()
                    raise
                else:
                    await transaction.commit()

    async def save(
        self,
        user_uuid: str,
        propositions: list[Proposition],
        embeddings: list[list[float]],
        arguments: list[Argument],
        *,
        source_uuid: str | None,
        target_unit_uuid: str | None,
        embedding_model: str,
        existing_proposition_uuids: list[str],
        existing_argument_uuids: list[str],
        transaction: AsyncTransaction | None = None,
    ) -> list[Argument]:
        """Atomically persist new knowledge and optional source provenance."""
        edge_created_at = Edge().created_at
        proposition_rows = [
            {
                'uuid': proposition.uuid,
                'name': proposition.name,
                'statement': proposition.statement,
                'embedding': embedding,
                'created_at': proposition.created_at,
                'updated_at': proposition.updated_at,
                'invalidated_at': proposition.invalidated_at,
            }
            for proposition, embedding in zip(
                propositions, embeddings, strict=True
            )
        ]

        async def persist(transaction: AsyncTransaction) -> list[Argument]:
            origin = 'source' if source_uuid is not None else 'constructed'
            result = await transaction.run(
                READ_OWNED_ARGUMENTS, user_uuid=user_uuid
            )
            existing_records = await result.data()
            identity_map = {
                (
                    row['conclusion_uuid'],
                    frozenset(row['premise_uuids']),
                ): Argument(
                    uuid=row['argument']['uuid'],
                    premise_uuids=row['premise_uuids'],
                    conclusion_uuid=row['conclusion_uuid'],
                    created_at=native_datetime(row['argument']['created_at']),
                    updated_at=native_datetime(row['argument']['updated_at']),
                    invalidated_at=(
                        native_datetime(row['argument']['invalidated_at'])
                        if row['argument']['invalidated_at'] is not None
                        else None
                    ),
                )
                for row in existing_records
            }
            stored_arguments = []
            argument_rows = []
            for argument in arguments:
                identity = (
                    argument.conclusion_uuid,
                    frozenset(argument.premise_uuids),
                )
                stored = identity_map.get(identity)
                if stored is None:
                    stored = argument
                    identity_map[identity] = stored
                    argument_rows.append(
                        {
                            'uuid': argument.uuid,
                            'premise_uuids': argument.premise_uuids,
                            'conclusion_uuid': argument.conclusion_uuid,
                            'created_at': argument.created_at,
                            'updated_at': argument.updated_at,
                            'invalidated_at': argument.invalidated_at,
                            'origin': origin,
                        }
                    )
                stored_arguments.append(stored)

            if source_uuid is not None:
                result = await transaction.run(
                    REQUIRE_OWNED_SOURCE_UNIT,
                    user_uuid=user_uuid,
                    source_uuid=source_uuid,
                    target_unit_uuid=target_unit_uuid,
                )
                record = await result.single()
                if record['count'] != 1:
                    raise PermissionError('Source unit is not owned')
            if proposition_rows:
                result = await transaction.run(
                    SAVE_PROPOSITIONS,
                    user_uuid=user_uuid,
                    propositions=proposition_rows,
                    embedding_model=embedding_model,
                    edge_created_at=edge_created_at,
                )
                record = await result.single()
                if record['count'] != len(proposition_rows):
                    raise PermissionError('Knowledge owner does not exist')
            if argument_rows:
                result = await transaction.run(
                    SAVE_ARGUMENTS,
                    user_uuid=user_uuid,
                    arguments=argument_rows,
                    edge_created_at=edge_created_at,
                )
                record = await result.single()
                if record['count'] != len(argument_rows):
                    raise PermissionError('Argument endpoints are not owned')
            if source_uuid is not None:
                proposition_uuids = list(
                    dict.fromkeys(
                        [row['uuid'] for row in proposition_rows]
                        + existing_proposition_uuids
                    )
                )
                if proposition_uuids:
                    result = await transaction.run(
                        ATTACH_SOURCE_PROVENANCE,
                        user_uuid=user_uuid,
                        source_uuid=source_uuid,
                        proposition_uuids=proposition_uuids,
                        edge_created_at=edge_created_at,
                    )
                    record = await result.single()
                    if record['count'] != len(proposition_uuids):
                        raise PermissionError(
                            'Source or proposition is not owned'
                        )
                argument_uuids = list(
                    dict.fromkeys(
                        [argument.uuid for argument in stored_arguments]
                        + existing_argument_uuids
                    )
                )
                if argument_uuids:
                    result = await transaction.run(
                        ATTACH_ARGUMENT_PROVENANCE,
                        user_uuid=user_uuid,
                        source_uuid=source_uuid,
                        argument_uuids=argument_uuids,
                        edge_created_at=edge_created_at,
                    )
                    record = await result.single()
                    if record['count'] != len(argument_uuids):
                        raise PermissionError('Source or argument is not owned')
            return stored_arguments

        if transaction is not None:
            return await persist(transaction)
        async with self.serialized(user_uuid) as owned_transaction:
            return await persist(owned_transaction)

    async def search(
        self,
        user_uuid: str,
        embedding: list[float],
        *,
        limit: int,
        embedding_model: str,
    ) -> list[Proposition]:
        """Return exact owner-scoped cosine matches in stable score order."""
        async with self._session_factory() as session:
            result = await session.run(
                SEARCH_PROPOSITIONS,
                user_uuid=user_uuid,
                embedding=embedding,
                embedding_model=embedding_model,
            )
            record = await result.single()
        if record['incompatible_uuids']:
            raise ValueError(
                f'Incompatible embedding model for propositions: {record["incompatible_uuids"]}'
            )
        ranked = sorted(
            record['ranked'],
            key=lambda item: (-item['score'], item['proposition']['uuid']),
        )
        return [_proposition(item['proposition']) for item in ranked[:limit]]

    async def context(
        self,
        user_uuid: str,
        matched: list[Proposition],
        *,
        argument_limit: int,
    ) -> KnowledgeContext:
        """Load bounded owned arguments and all their proposition endpoints."""
        matched_uuids = [proposition.uuid for proposition in matched]
        async with self._session_factory() as session:
            result = await session.run(
                READ_MATCHED_PROPOSITIONS,
                user_uuid=user_uuid,
                matched_uuids=matched_uuids,
            )
            matched_rows = await result.data()
            result = await session.run(
                READ_CONTEXT,
                user_uuid=user_uuid,
                matched_uuids=matched_uuids,
                argument_limit=argument_limit,
            )
            rows = await result.data()
        arguments = [
            Argument(
                uuid=row['argument']['uuid'],
                created_at=native_datetime(row['argument']['created_at']),
                updated_at=native_datetime(row['argument']['updated_at']),
                invalidated_at=(
                    native_datetime(row['argument']['invalidated_at'])
                    if row['argument']['invalidated_at'] is not None
                    else None
                ),
                premise_uuids=row['argument']['premise_uuids'],
                conclusion_uuid=row['argument']['conclusion_uuid'],
            )
            for row in rows
        ]
        owned_matched = [
            _proposition(row['proposition']) for row in matched_rows
        ]
        propositions_by_uuid = {
            proposition.uuid: proposition for proposition in owned_matched
        }
        propositions_by_uuid.update(
            {
                proposition['uuid']: _proposition(proposition)
                for row in rows
                for proposition in row['propositions']
            }
        )
        return KnowledgeContext(
            matched=owned_matched,
            propositions=sorted(
                propositions_by_uuid.values(), key=lambda item: item.uuid
            ),
            arguments=arguments,
        )


def _proposition(row: dict) -> Proposition:
    return Proposition(
        uuid=row['uuid'],
        name=row['name'],
        statement=row['statement'],
        created_at=native_datetime(row['created_at']),
        updated_at=native_datetime(row['updated_at']),
        invalidated_at=(
            native_datetime(row['invalidated_at'])
            if row['invalidated_at'] is not None
            else None
        ),
    )
