"""Persistence access for source facts, triplets, and triplet hubs."""

from collections.abc import Callable

from kms2.core.model import (
    SourceEntity,
    SourceEvent,
    SourceFact,
    SourceTripletHub,
    SourceTripletHubGroup,
    SourceTripletOccurrence,
)
from kms2.database.semantic.queries.source_triplet import (
    CLEAR_SOURCE_FACTS_AND_TRIPLETS,
    READ_SOURCE_TRIPLET_HUB_GROUPS,
    REPLACE_SOURCE_FACTS_AND_TRIPLETS,
    REPLACE_SOURCE_TRIPLET_HUBS,
)


class SourceTripletRepository:
    """Access persisted source facts, triplets, and triplet hubs."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def replace_source_facts_and_triplets(
        self,
        source_uuid: str,
        source_facts: list[SourceFact],
        triplet_occurrences: list[SourceTripletOccurrence],
    ) -> None:
        """Replace all source facts and their decomposed triplets."""
        if not source_facts:
            async with self._session_factory() as session:
                result = await session.run(
                    CLEAR_SOURCE_FACTS_AND_TRIPLETS,
                    source_uuid=source_uuid,
                )
                await result.consume()
            return

        parameters = {
            'source_uuid': source_uuid,
            'source_facts': [
                {
                    'uuid': fact.uuid,
                    'source_uuid': fact.source_uuid,
                    'source_block_uuid': fact.source_block_uuid,
                    'text': fact.text,
                }
                for fact in source_facts
            ],
            'triplets': [
                {
                    'uuid': occurrence.triplet.uuid,
                    'source_uuid': occurrence.triplet.source_uuid,
                    'source_block_uuid': occurrence.triplet.source_block_uuid,
                    'subject_uuid': occurrence.triplet.subject_uuid,
                    'object_uuid': occurrence.triplet.object_uuid,
                    'predicate_uuid': occurrence.triplet.predicate_uuid,
                }
                for occurrence in triplet_occurrences
            ],
            'source_entities': [
                _endpoint_row(occurrence.subject)
                for occurrence in triplet_occurrences
                if isinstance(occurrence.subject, SourceEntity)
            ]
            + [
                _endpoint_row(occurrence.object)
                for occurrence in triplet_occurrences
                if isinstance(occurrence.object, SourceEntity)
            ],
            'source_events': [
                _endpoint_row(occurrence.subject)
                for occurrence in triplet_occurrences
                if isinstance(occurrence.subject, SourceEvent)
            ]
            + [
                _endpoint_row(occurrence.object)
                for occurrence in triplet_occurrences
                if isinstance(occurrence.object, SourceEvent)
            ],
            'source_predicates': [
                {
                    'uuid': occurrence.predicate.uuid,
                    'source_uuid': occurrence.predicate.source_uuid,
                    'source_block_uuid': occurrence.predicate.source_block_uuid,
                    'predicate': occurrence.predicate.predicate,
                }
                for occurrence in triplet_occurrences
            ],
            'fact_triplet_pairs': [
                {
                    'source_fact_uuid': occurrence.fact.uuid,
                    'triplet_uuid': occurrence.triplet.uuid,
                }
                for occurrence in triplet_occurrences
            ],
        }
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_FACTS_AND_TRIPLETS,
                **parameters,
            )
            await result.consume()

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
