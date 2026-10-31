"""Persistence access for source entity graphs."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.block import SourceBlock
from kms2.core.model.similarity import SourceEntityHubSearchMatch
from kms2.core.model.source_extraction import (
    SourceEvidenceInput,
    SourceExtractionTarget,
)
from kms2.core.model.source_semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionResult,
)
from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubMember,
)
from kms2.core.windowing import project_block
from kms2.database.source.queries.source_entity import (
    READ_SOURCE_ENTITIES,
    READ_SOURCE_ENTITY_HUB_CANDIDATES,
    REPLACE_SOURCE_ENTITIES,
    REPLACE_SOURCE_ENTITY_HUBS,
    SEARCH_SOURCE_ENTITY_HUBS,
    UPDATE_SOURCE_ENTITY_DESCRIPTION,
)


class SourceEntityRepository:
    """Access persisted source entity semantic graph data."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace_source_entities(
        self, source_uuid: str, entities: list[SourceEntity]
    ) -> None:
        """Replace one source's raw entity occurrences and owned hub graph."""
        rows = [
            {
                'uuid': entity.uuid,
                'name': entity.name,
                'target_block_uuid': entity.target.source_blocks[0].uuid,
            }
            for entity in entities
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_ENTITIES,
                source_uuid=source_uuid,
                entities=rows,
            )
            await result.single(strict=True)

    async def load_source_entities(
        self, source_uuid: str
    ) -> list[SourceEntity]:
        """Load raw entity occurrences with ordered block evidence."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITIES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceEntity(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                name=row['name'],
                target=SourceExtractionTarget(
                    source_blocks=[
                        _source_block(block) for block in row['target_blocks']
                    ],
                ),
            )
            for row in rows
        ]

    async def update_source_entity_description(
        self,
        source_uuid: str,
        results: list[SourceEntityDescriptionResult],
    ) -> None:
        """Update descriptions and embeddings on existing entity nodes."""
        await self._update_description(
            UPDATE_SOURCE_ENTITY_DESCRIPTION,
            source_uuid,
            results,
        )

    async def read_source_entity_hub_candidates(
        self,
        source_uuid: str,
        groups: list[list[SourceEntityHubMember]],
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourceEntityHubCandidate]:
        """Retrieve the best member pair for each top-scoring source-local group pair."""
        group_rows = [
            {
                'group_uuid': min(member.uuid for member in group),
                'member_uuids': [member.uuid for member in group],
            }
            for group in groups
        ]
        members = {member.uuid: member for group in groups for member in group}
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITY_HUB_CANDIDATES,
                source_uuid=source_uuid,
                groups=group_rows,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        candidates = []
        for row in rows:
            left = members[row['left_uuid']]
            right = members[row['right_uuid']]
            candidates.append(
                SourceEntityHubCandidate(
                    left_group_uuid=row['left_group_uuid'],
                    right_group_uuid=row['right_group_uuid'],
                    left_uuid=left.uuid,
                    left_name=left.name,
                    left_description=left.description,
                    left_evidence=left.evidence,
                    right_uuid=right.uuid,
                    right_name=right.name,
                    right_description=right.description,
                    right_evidence=right.evidence,
                    score=row['score'],
                )
            )
        return candidates

    async def read_source_entity_hub_members(
        self, source_uuid: str
    ) -> list[SourceEntityHubMember]:
        """Read source entities with UUID-free owning evidence."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITIES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceEntityHubMember(
                uuid=row['uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
                description=row['description'],
                evidence=SourceEvidenceInput(
                    target_blocks=[
                        project_block(_source_block(block))
                        for block in row['target_blocks']
                    ],
                ),
            )
            for row in rows
        ]

    async def search_source_entity_hubs(
        self,
        source_uuid: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SourceEntityHubSearchMatch]:
        """Search source hubs via their best matching embedded member."""
        _validate_search(query_embedding, top_k)
        async with self._session_factory() as session:
            result = await session.run(
                SEARCH_SOURCE_ENTITY_HUBS,
                source_uuid=source_uuid,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            rows = await result.data()
        return [SourceEntityHubSearchMatch.model_validate(row) for row in rows]

    async def replace_source_entity_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceEntityHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace identity-only source hubs and their occurrence memberships."""
        rows = [
            {
                **hub.model_dump(),
                'member_uuids': member_uuids,
            }
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_ENTITY_HUBS,
                source_uuid=source_uuid,
                hubs=rows,
            )
            await result.single(strict=True)

    async def _update_description(
        self,
        query: str,
        source_uuid: str,
        results: list[SourceEntityDescriptionResult],
    ) -> None:
        if not results:
            return
        rows = [
            {
                'uuid': result.uuid,
                'description': result.description,
                'embedding': result.embedding,
            }
            for result in results
        ]
        async with self._session_factory() as session:
            response = await session.run(
                query,
                source_uuid=source_uuid,
                rows=rows,
            )
            await response.consume()


def _source_block(row: dict[str, object]) -> SourceBlock:
    """Build source evidence returned by an ordered block-chain projection."""
    return SourceBlock(
        uuid=row['uuid'],
        block_type=row['block_type'],
        content=row['content'],
    )


def _validate_search(embedding: list[float], top_k: int) -> None:
    if not embedding:
        raise ValueError('query_embedding must not be empty')
    if top_k <= 0:
        raise ValueError('top_k must be positive')
