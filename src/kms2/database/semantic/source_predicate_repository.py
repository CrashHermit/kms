"""Persistence access for source predicate graphs."""

from collections.abc import Callable

from kms2.core.model import (
    SourcePredicate,
    SourcePredicateDescriptionResult,
    SourcePredicateHub,
    SourcePredicateHubCandidate,
    SourcePredicateHubMember,
    SourcePredicateSimilarityMatch,
)
from kms2.database.semantic.queries.source_predicate import (
    DETECT_SOURCE_PREDICATE_COMMUNITIES,
    DROP_SOURCE_PREDICATE_HUB_GRAPH,
    FIND_SIMILAR_SOURCE_PREDICATES,
    READ_SOURCE_PREDICATE_HUB_CANDIDATES,
    READ_SOURCE_PREDICATES,
    REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES,
    REPLACE_SOURCE_PREDICATE_HUBS,
    UPDATE_SOURCE_PREDICATE_DESCRIPTION,
)


class SourcePredicateRepository:
    """Access persisted source predicate semantic graph data."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def load_source_predicates(
        self, source_uuid: str
    ) -> list[SourcePredicate]:
        """Load raw predicate occurrences for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PREDICATES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourcePredicate(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                predicate=row['predicate'],
            )
            for row in rows
        ]

    async def update_source_predicate_description(
        self,
        source_uuid: str,
        results: list[SourcePredicateDescriptionResult],
    ) -> None:
        """Update descriptions and embeddings on existing predicate nodes."""
        await self._update_description(
            UPDATE_SOURCE_PREDICATE_DESCRIPTION,
            source_uuid,
            results,
        )

    async def find_similar_source_predicates(
        self,
        source_predicate_uuid: str,
        *,
        top_k: int,
    ) -> list[SourcePredicateSimilarityMatch]:
        """Find nearest source predicates using Neo4j's predicate index."""
        rows = await self._find_similar(
            FIND_SIMILAR_SOURCE_PREDICATES,
            source_predicate_uuid,
            top_k,
        )
        return [
            SourcePredicateSimilarityMatch(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                predicate=row['predicate'],
                description=row['description'],
                score=row['score'],
            )
            for row in rows
        ]

    async def read_source_predicate_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourcePredicateHubCandidate]:
        """Read canonical same-source predicate pairs for filtering."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PREDICATE_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [SourcePredicateHubCandidate.model_validate(row) for row in rows]

    async def replace_source_predicate_accepted_edges(
        self,
        source_uuid: str,
        pairs: list[SourcePredicateHubCandidate],
    ) -> None:
        """Replace accepted predicate similarity edges for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_source_predicate_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourcePredicateHubMember]]:
        """Detect overlapping communities among source predicate occurrences."""
        graph_name = f'kms2-source-predicate-hubs-{source_uuid}'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_SOURCE_PREDICATE_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_SOURCE_PREDICATE_COMMUNITIES,
                    source_uuid=source_uuid,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[SourcePredicateHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        SourcePredicateHubMember(
                            uuid=row['uuid'],
                            predicate=row['predicate'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_SOURCE_PREDICATE_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def replace_source_predicate_hubs(
        self,
        source_uuid: str,
        hubs: list[SourcePredicateHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace source-local predicate hubs and memberships."""
        rows = [
            {
                **hub.model_dump(),
                'member_uuids': member_uuids,
            }
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PREDICATE_HUBS,
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
        results: list[SourcePredicateDescriptionResult],
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
