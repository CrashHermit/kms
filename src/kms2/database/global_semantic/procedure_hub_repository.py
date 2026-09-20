"""Persistence access for global procedure hub graphs."""

from collections.abc import Callable

from kms2.core.model import (
    GlobalProcedureHub,
    GlobalProcedureHubCandidate,
    GlobalProcedureHubMember,
)
from kms2.database.global_semantic.queries.procedure_hub import (
    DETECT_GLOBAL_PROCEDURE_HUB_COMMUNITIES,
    DROP_GLOBAL_PROCEDURE_HUB_GRAPH,
    READ_GLOBAL_PROCEDURE_HUB_CANDIDATES,
    REPLACE_GLOBAL_PROCEDURE_HUB_ACCEPTED_EDGES,
    REPLACE_GLOBAL_PROCEDURE_HUBS,
)


class GlobalProcedureHubRepository:
    """Access persisted source-hub relationships and global hubs."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def read_global_procedure_hub_candidates(
        self,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[GlobalProcedureHubCandidate]:
        """Read cross-source source-procedure-hub vector candidates."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_GLOBAL_PROCEDURE_HUB_CANDIDATES,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [GlobalProcedureHubCandidate.model_validate(row) for row in rows]

    async def replace_global_procedure_hub_accepted_edges(
        self, pairs: list[GlobalProcedureHubCandidate]
    ) -> None:
        """Replace derived cross-source source-hub similarity edges."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_GLOBAL_PROCEDURE_HUB_ACCEPTED_EDGES,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_global_procedure_hub_communities(
        self,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[GlobalProcedureHubMember]]:
        """Detect cross-source communities among source procedure hubs."""
        graph_name = 'kms2-global-procedure-hubs'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_GLOBAL_PROCEDURE_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_GLOBAL_PROCEDURE_HUB_COMMUNITIES,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[GlobalProcedureHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        GlobalProcedureHubMember(
                            uuid=row['uuid'],
                            canonical_name=row['canonical_name'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_GLOBAL_PROCEDURE_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def replace_global_procedure_hubs(
        self,
        hubs: list[GlobalProcedureHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace global hubs and relationship-based memberships."""
        rows = [
            {**hub.model_dump(), 'member_uuids': member_uuids}
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(REPLACE_GLOBAL_PROCEDURE_HUBS, hubs=rows)
            await result.consume()


__all__ = ['GlobalProcedureHubRepository']
