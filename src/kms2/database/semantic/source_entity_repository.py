"""Persistence access for source entity graphs."""

from collections.abc import Callable

from kms2.core.model import (
    SourceEntity,
    SourceEntityDescriptionResult,
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubMember,
    SourceEntitySimilarityMatch,
)
from kms2.database.semantic.queries.source_entity import (
    DETECT_SOURCE_ENTITY_COMMUNITIES,
    DROP_SOURCE_ENTITY_HUB_GRAPH,
    FIND_SIMILAR_SOURCE_ENTITIES,
    READ_SOURCE_ENTITIES,
    READ_SOURCE_ENTITY_HUB_CANDIDATES,
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES,
    REPLACE_SOURCE_ENTITY_HUBS,
    UPDATE_SOURCE_ENTITY_DESCRIPTION,
)


class SourceEntityRepository:
    """Access persisted source entity semantic graph data."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def load_source_entities(
        self, source_uuid: str
    ) -> list[SourceEntity]:
        """Load raw entity occurrences for one source."""
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
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
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

    async def find_similar_source_entities(
        self,
        source_entity_uuid: str,
        *,
        top_k: int,
    ) -> list[SourceEntitySimilarityMatch]:
        """Find nearest source entities using Neo4j's entity index."""
        rows = await self._find_similar(
            FIND_SIMILAR_SOURCE_ENTITIES,
            source_entity_uuid,
            top_k,
        )
        return [
            SourceEntitySimilarityMatch(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
                description=row['description'],
                score=row['score'],
            )
            for row in rows
        ]

    async def read_source_entity_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourceEntityHubCandidate]:
        """Read canonical same-source entity pairs for filtering."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITY_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [SourceEntityHubCandidate.model_validate(row) for row in rows]

    async def replace_source_entity_accepted_edges(
        self,
        source_uuid: str,
        pairs: list[SourceEntityHubCandidate],
    ) -> None:
        """Replace accepted entity similarity edges for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_source_entity_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourceEntityHubMember]]:
        """Detect overlapping communities among source entity occurrences."""
        graph_name = f'kms2-source-entity-hubs-{source_uuid}'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_SOURCE_ENTITY_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_SOURCE_ENTITY_COMMUNITIES,
                    source_uuid=source_uuid,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[SourceEntityHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        SourceEntityHubMember(
                            uuid=row['uuid'],
                            name=row['name'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_SOURCE_ENTITY_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def replace_source_entity_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceEntityHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace source-local entity hubs and memberships."""
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
            await result.consume()

    async def _find_similar(
        self,
        query: str,
        query_uuid: str,
        top_k: int,
    ) -> list[dict[str, object]]:
        async with self._session_factory() as session:
            result = await session.run(
                query,
                query_uuid=query_uuid,
                top_k=top_k,
                candidate_limit=top_k + 1,
            )
            return await result.data()

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
