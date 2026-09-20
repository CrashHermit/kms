"""Persistence access for global statement hub graphs."""

from collections.abc import Callable

from kms2.core.model import (
    GlobalStatementHub,
    GlobalStatementHubCandidate,
    GlobalStatementHubMember,
)
from kms2.database.global_semantic.queries.statement_hub import (
    DETECT_GLOBAL_STATEMENT_HUB_COMMUNITIES,
    DROP_GLOBAL_STATEMENT_HUB_GRAPH,
    READ_GLOBAL_STATEMENT_HUB_CANDIDATES,
    REPLACE_GLOBAL_STATEMENT_HUB_ACCEPTED_EDGES,
    REPLACE_GLOBAL_STATEMENT_HUBS,
)


class GlobalStatementHubRepository:
    """Access persisted source-hub relationships and global hubs."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def read_global_statement_hub_candidates(
        self,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[GlobalStatementHubCandidate]:
        """Read cross-source source-statement-hub vector candidates."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_GLOBAL_STATEMENT_HUB_CANDIDATES,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [GlobalStatementHubCandidate.model_validate(row) for row in rows]

    async def replace_global_statement_hub_accepted_edges(
        self, pairs: list[GlobalStatementHubCandidate]
    ) -> None:
        """Replace derived cross-source source-hub similarity edges."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_GLOBAL_STATEMENT_HUB_ACCEPTED_EDGES,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_global_statement_hub_communities(
        self,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[GlobalStatementHubMember]]:
        """Detect cross-source communities among source statement hubs."""
        graph_name = 'kms2-global-statement-hubs'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_GLOBAL_STATEMENT_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_GLOBAL_STATEMENT_HUB_COMMUNITIES,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[GlobalStatementHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        GlobalStatementHubMember(
                            uuid=row['uuid'],
                            canonical_name=row['canonical_name'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_GLOBAL_STATEMENT_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def replace_global_statement_hubs(
        self,
        hubs: list[GlobalStatementHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace global hubs and relationship-based memberships."""
        rows = [
            {**hub.model_dump(), 'member_uuids': member_uuids}
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(REPLACE_GLOBAL_STATEMENT_HUBS, hubs=rows)
            await result.consume()


__all__ = ['GlobalStatementHubRepository']
