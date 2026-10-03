"""Persistence access for global triplet projections and hubs."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHub,
    GlobalTripletHubGroup,
)
from kms2.database.global_semantic.queries.global_triplet import (
    READ_GLOBAL_TRIPLET_HUB_GROUPS,
    REPLACE_GLOBAL_TRIPLET_HUBS,
    REPLACE_GLOBAL_TRIPLETS,
)


class GlobalTripletRepository:
    """Access projected global triplets and global triplet hubs."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace_global_triplets(self) -> int:
        """Replace projected global triplet evidence occurrences."""
        async with self._session_factory() as session:
            result = await session.run(REPLACE_GLOBAL_TRIPLETS)
            rows = await result.data()
        return rows[0]['persisted']

    async def read_global_triplet_hub_groups(
        self,
    ) -> list[GlobalTripletHubGroup]:
        """Read deterministic exact global triplet role groups."""
        async with self._session_factory() as session:
            result = await session.run(READ_GLOBAL_TRIPLET_HUB_GROUPS)
            rows = await result.data()
        return [_global_triplet_hub_group(row) for row in rows]

    async def replace_global_triplet_hubs(
        self,
        hubs: list[GlobalTripletHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace global triplet hubs and projected-triplet memberships."""
        rows = [
            {
                **hub.model_dump(),
                'global_triplet_uuids': global_triplet_uuids,
            }
            for hub, global_triplet_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(REPLACE_GLOBAL_TRIPLET_HUBS, hubs=rows)
            await result.consume()


def _global_triplet_hub_group(row: dict) -> GlobalTripletHubGroup:
    """Normalize Neo4j collections before global group validation."""
    evidence = sorted(
        row['evidence'],
        key=lambda item: (
            item['global_triplet_uuid'],
            item['source_triplet_hub_uuid'],
        ),
    )
    return GlobalTripletHubGroup.model_validate(
        {
            **dict(row),
            'global_triplet_uuids': sorted(row['global_triplet_uuids']),
            'evidence': evidence,
            'subject_hub': {
                'name': row['subject_hub_name'],
                'description': row['subject_hub_description'],
            },
            'predicate_hub': {
                'name': row['predicate_hub_name'],
                'description': row['predicate_hub_description'],
            },
            'object_hub': {
                'name': row['object_hub_name'],
                'description': row['object_hub_description'],
            },
        }
    )
