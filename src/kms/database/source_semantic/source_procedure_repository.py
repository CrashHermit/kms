"""Persistence access for source procedure graphs."""

from collections.abc import Callable, Sequence

from neo4j import AsyncSession

from kms.core.model.source_semantic.source_procedure import (
    SourceProcedure,
)
from kms.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHub,
    SourceProcedureHubCandidate,
    SourceProcedureHubMember,
)
from kms.database.source_semantic.queries.source_procedure import (
    DETECT_SOURCE_PROCEDURE_COMMUNITIES,
    DROP_SOURCE_PROCEDURE_HUB_GRAPH,
    READ_SOURCE_PROCEDURE_HUB_CANDIDATES,
    READ_SOURCE_PROCEDURES,
    REPLACE_SOURCE_PROCEDURE_ACCEPTED_EDGES,
    REPLACE_SOURCE_PROCEDURE_HUBS,
)


class SourceProcedureRepository:
    """Access persisted source procedure semantic graph data."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_source_procedures(
        self, source_uuid: str
    ) -> list[SourceProcedure]:
        """Load persisted procedures for source-local hub processing."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PROCEDURES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceProcedure(
                uuid=row['uuid'],
                member_block_uuids=row['member_block_uuids'],
                content=row['content'],
                embedding=row['embedding'],
            )
            for row in rows
        ]

    async def read_source_procedure_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourceProcedureHubCandidate]:
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PROCEDURE_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            return [
                SourceProcedureHubCandidate.model_validate(row)
                for row in await result.data()
            ]

    async def replace_source_procedure_accepted_edges(
        self, source_uuid: str, pairs: list[SourceProcedureHubCandidate]
    ) -> None:
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PROCEDURE_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_source_procedure_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourceProcedureHubMember]]:
        if not await self.load_source_procedures(source_uuid):
            return []
        return await self._detect_statement_or_procedure_communities(
            source_uuid,
            max_iterations=max_iterations,
            min_association_strength=min_association_strength,
            minimum_community_size=minimum_community_size,
            query=DETECT_SOURCE_PROCEDURE_COMMUNITIES,
            drop_query=DROP_SOURCE_PROCEDURE_HUB_GRAPH,
            graph_name=f'kms2-source-procedure-hubs-{source_uuid}',
            model=SourceProcedureHubMember,
        )

    async def replace_source_procedure_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceProcedureHub],
        memberships: list[list[str]],
    ) -> None:
        await self._replace_hubs(
            REPLACE_SOURCE_PROCEDURE_HUBS, source_uuid, hubs, memberships
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
        model: type[SourceProcedureHubMember],
    ) -> list[list[SourceProcedureHubMember]]:
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
        self,
        query: str,
        source_uuid: str,
        hubs: Sequence[SourceProcedureHub],
        memberships: Sequence[list[str]],
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
