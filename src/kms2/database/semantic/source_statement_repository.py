"""Persistence access for source statement graphs."""

from collections.abc import Callable

from kms2.core.model import (
    SourceStatement,
    SourceStatementDescriptionResult,
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubMember,
)
from kms2.database.semantic.queries.source_statement import (
    DETECT_SOURCE_STATEMENT_COMMUNITIES,
    DROP_SOURCE_STATEMENT_HUB_GRAPH,
    READ_SOURCE_STATEMENT_HUB_CANDIDATES,
    READ_SOURCE_STATEMENTS,
    REPLACE_SOURCE_STATEMENT_ACCEPTED_EDGES,
    REPLACE_SOURCE_STATEMENT_HUBS,
    UPDATE_SOURCE_STATEMENT_DESCRIPTION,
)


class SourceStatementRepository:
    """Access persisted source statement semantic graph data."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def load_source_statements(
        self, source_uuid: str
    ) -> list[SourceStatement]:
        """Load source-scoped statement pointers for semantic description."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_STATEMENTS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceStatement(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                member_block_uuids=row['member_block_uuids'],
                is_exercise=row['is_exercise'],
            )
            for row in rows
        ]

    async def update_source_statement_description(
        self,
        source_uuid: str,
        results: list[SourceStatementDescriptionResult],
    ) -> None:
        """Update descriptions and embeddings on source statement nodes."""
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
                UPDATE_SOURCE_STATEMENT_DESCRIPTION,
                source_uuid=source_uuid,
                rows=rows,
            )
            await response.consume()

    async def read_source_statement_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourceStatementHubCandidate]:
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_STATEMENT_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            return [
                SourceStatementHubCandidate.model_validate(row)
                for row in await result.data()
            ]

    async def replace_source_statement_accepted_edges(
        self, source_uuid: str, pairs: list[SourceStatementHubCandidate]
    ) -> None:
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_STATEMENT_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_source_statement_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourceStatementHubMember]]:
        return await self._detect_statement_or_procedure_communities(
            source_uuid,
            max_iterations=max_iterations,
            min_association_strength=min_association_strength,
            minimum_community_size=minimum_community_size,
            query=DETECT_SOURCE_STATEMENT_COMMUNITIES,
            drop_query=DROP_SOURCE_STATEMENT_HUB_GRAPH,
            graph_name=f'kms2-source-statement-hubs-{source_uuid}',
            model=SourceStatementHubMember,
        )

    async def replace_source_statement_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceStatementHub],
        memberships: list[list[str]],
    ) -> None:
        await self._replace_hubs(
            REPLACE_SOURCE_STATEMENT_HUBS, source_uuid, hubs, memberships
        )

    async def _detect_statement_or_procedure_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
        query: str,
        drop_query: str,
        graph_name: str,
        model,
    ) -> list[list[object]]:
        async with self._session_factory() as session:
            await (
                await session.run(drop_query, graph_name=graph_name)
            ).consume()
            try:
                result = await session.run(
                    query,
                    source_uuid=source_uuid,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
            finally:
                await (
                    await session.run(drop_query, graph_name=graph_name)
                ).consume()
        communities = {}
        for row in rows:
            communities.setdefault(row['community_id'], []).append(
                model.model_validate(row)
            )
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def _replace_hubs(
        self, query: str, source_uuid: str, hubs, memberships
    ) -> None:
        rows = [
            {**hub.model_dump(), 'member_uuids': member_uuids}
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                query, source_uuid=source_uuid, hubs=rows
            )
            await result.consume()
