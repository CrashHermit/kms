"""Persistence access for node-first source facts and evidence pointers."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.block import SourceBlock
from kms2.core.model.source_semantic.source_fact_extraction import (
    SourceFact,
    SourceFactContext,
    SourceFactTarget,
)
from kms2.database.source_semantic.queries.source_fact import (
    READ_SOURCE_FACT_CONTEXTS,
    READ_SOURCE_FACTS,
    REPLACE_SOURCE_FACTS,
)


class SourceFactRepository:
    """Access durable source facts and their evidence pointer nodes."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace_source_facts(
        self, source_uuid: str, source_facts: list[SourceFact]
    ) -> None:
        """Replace one source's facts and node-first evidence graph."""
        rows = [
            {
                'uuid': fact.uuid,
                'text': fact.text,
                'target_uuid': fact.target.uuid,
                'context_before_uuid': fact.context_before.uuid,
                'context_after_uuid': fact.context_after.uuid,
                'target_block_uuids': [
                    block.uuid for block in fact.target.source_blocks
                ],
                'context_before_block_uuids': [
                    block.uuid for block in fact.context_before.source_blocks
                ],
                'context_after_block_uuids': [
                    block.uuid for block in fact.context_after.source_blocks
                ],
            }
            for fact in source_facts
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_FACTS,
                source_uuid=source_uuid,
                facts=rows,
            )
            await result.consume()

    async def load_source_facts(self, source_uuid: str) -> list[SourceFact]:
        """Read facts and reconstruct their canonical evidence pointers."""
        async with self._session_factory() as session:
            fact_result = await session.run(
                READ_SOURCE_FACTS,
                source_uuid=source_uuid,
            )
            fact_rows = await fact_result.data()
            context_result = await session.run(
                READ_SOURCE_FACT_CONTEXTS,
                source_uuid=source_uuid,
            )
            context_rows = await context_result.data()
        contexts: dict[str, dict[str, SourceFactContext]] = {}
        for row in context_rows:
            contexts.setdefault(row['fact_uuid'], {})[row['context_kind']] = (
                SourceFactContext(
                    uuid=row['context_uuid'],
                    source_blocks=[
                        _source_block(block) for block in row['blocks']
                    ],
                )
            )
        return [
            SourceFact(
                uuid=row['uuid'],
                text=row['text'],
                target=SourceFactTarget(
                    uuid=row['target_uuid'],
                    source_blocks=[
                        _source_block(block) for block in row['target_blocks']
                    ],
                ),
                context_before=contexts[row['uuid']]['HAS_CONTEXT_BEFORE'],
                context_after=contexts[row['uuid']]['HAS_CONTEXT_AFTER'],
            )
            for row in fact_rows
        ]


def _source_block(row: dict[str, object]) -> SourceBlock:
    """Build the relationship material returned by a fact query."""
    return SourceBlock(
        uuid=row['uuid'],
        block_type=row['block_type'],
        content=row['content'],
    )
