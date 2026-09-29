"""Persistence access for source triplets and triplet hubs."""

from collections.abc import Callable

from kms2.core.model.source_semantic.source_entity import SourceEntity
from kms2.core.model.source_semantic.source_event import SourceEvent
from kms2.core.model.source_semantic.source_triplet import (
    SourceTripletOccurrence,
)
from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHub,
    SourceTripletHubGroup,
)
from kms2.database.source_semantic.queries.source_triplet import (
    READ_SOURCE_TRIPLET_HUB_GROUPS,
    REPLACE_SOURCE_TRIPLET_HUBS,
    REPLACE_SOURCE_TRIPLETS,
)
from kms2.database.source_semantic.source_fact_repository import (
    SourceFactRepository,
)


class SourceTripletRepository:
    """Access persisted source triplets and triplet hubs."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory
        self._fact_repository = SourceFactRepository(session_factory)

    async def replace_source_triplets(
        self,
        source_uuid: str,
        occurrences: list[SourceTripletOccurrence],
    ) -> None:
        """Replace only the triplets and typed occurrences for one source."""
        parameters = {
            'source_uuid': source_uuid,
            'triplets': [
                {
                    'uuid': occurrence.triplet.uuid,
                    'subject_uuid': occurrence.subject.uuid,
                    'object_uuid': occurrence.object.uuid,
                    'predicate_uuid': occurrence.predicate.uuid,
                }
                for occurrence in occurrences
            ],
            'source_entities': [
                _endpoint_row(occurrence.subject)
                for occurrence in occurrences
                if isinstance(occurrence.subject, SourceEntity)
            ]
            + [
                _endpoint_row(occurrence.object)
                for occurrence in occurrences
                if isinstance(occurrence.object, SourceEntity)
            ],
            'source_events': [
                _endpoint_row(occurrence.subject)
                for occurrence in occurrences
                if isinstance(occurrence.subject, SourceEvent)
            ]
            + [
                _endpoint_row(occurrence.object)
                for occurrence in occurrences
                if isinstance(occurrence.object, SourceEvent)
            ],
            'source_predicates': [
                {
                    'uuid': occurrence.predicate.uuid,
                    'source_uuid': occurrence.predicate.source_uuid,
                    'source_block_uuid': occurrence.predicate.source_block_uuid,
                    'predicate': occurrence.predicate.predicate,
                }
                for occurrence in occurrences
            ],
            'fact_triplet_pairs': [
                {
                    'source_fact_uuid': occurrence.fact.uuid,
                    'triplet_uuid': occurrence.triplet.uuid,
                }
                for occurrence in occurrences
            ],
        }
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_TRIPLETS,
                **parameters,
            )
            await result.consume()

    async def load_source_facts(self, source_uuid: str):
        """Load durable facts for the next source-semantic phase."""
        return await self._fact_repository.load_source_facts(source_uuid)

    async def read_source_triplet_hub_groups(
        self, source_uuid: str
    ) -> list[SourceTripletHubGroup]:
        """Read complete source-local triplet hub groups."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_TRIPLET_HUB_GROUPS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceTripletHubGroup.model_validate(
                {
                    **dict(row),
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
            for row in rows
        ]

    async def replace_source_triplet_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceTripletHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace source-local triplet hubs and raw-triplet memberships."""
        rows = [
            {
                **hub.model_dump(),
                'triplet_uuids': triplet_uuids,
            }
            for hub, triplet_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_TRIPLET_HUBS,
                source_uuid=source_uuid,
                hubs=rows,
            )
            await result.consume()


def _endpoint_row(endpoint: SourceEntity | SourceEvent) -> dict[str, str]:
    """Serialize one typed endpoint occurrence."""
    return {
        'uuid': endpoint.uuid,
        'source_uuid': endpoint.source_uuid,
        'source_block_uuid': endpoint.source_block_uuid,
        'name': endpoint.name,
    }
