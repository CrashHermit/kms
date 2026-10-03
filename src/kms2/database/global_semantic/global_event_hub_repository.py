"""Persistence access for global event hub graphs."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.global_semantic.global_event_hub import (
    GlobalEventHub,
    GlobalEventHubCandidate,
    GlobalEventHubMember,
)
from kms2.database.global_semantic.queries.global_event_hub import (
    DETECT_GLOBAL_EVENT_HUB_COMMUNITIES,
    DROP_GLOBAL_EVENT_HUB_GRAPH,
    READ_GLOBAL_EVENT_HUB_CANDIDATES,
    REPLACE_GLOBAL_EVENT_HUB_ACCEPTED_EDGES,
    REPLACE_GLOBAL_EVENT_HUBS,
)


class GlobalEventHubRepository:
    """Access persisted source-hub relationships and global hubs."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def read_global_event_hub_candidates(
        self,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[GlobalEventHubCandidate]:
        """Read cross-source source-event-hub vector candidates."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_GLOBAL_EVENT_HUB_CANDIDATES,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [GlobalEventHubCandidate.model_validate(row) for row in rows]

    async def replace_global_event_hub_accepted_edges(
        self, pairs: list[GlobalEventHubCandidate]
    ) -> None:
        """Replace derived cross-source source-hub similarity edges."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_GLOBAL_EVENT_HUB_ACCEPTED_EDGES,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_global_event_hub_communities(
        self,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[GlobalEventHubMember]]:
        """Detect cross-source communities among source event hubs."""
        graph_name = 'kms2-global-event-hubs'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_GLOBAL_EVENT_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_GLOBAL_EVENT_HUB_COMMUNITIES,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[GlobalEventHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        GlobalEventHubMember(
                            uuid=row['uuid'],
                            name=row['name'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_GLOBAL_EVENT_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        unique_communities = {
            tuple(sorted(member.uuid for member in members)): members
            for members in communities.values()
        }
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in unique_communities.values()
            if len(members) >= minimum_community_size
        ]

    async def replace_global_event_hubs(
        self,
        hubs: list[GlobalEventHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace global hubs and relationship-based memberships."""
        rows = [
            {**hub.model_dump(), 'member_uuids': member_uuids}
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(REPLACE_GLOBAL_EVENT_HUBS, hubs=rows)
            await result.consume()
