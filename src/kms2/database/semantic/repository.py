"""Persistence repository for raw semantic assertion occurrences."""

from collections.abc import Callable

from kms2.core.model import (
    Entity,
    EntityEnrichmentResult,
    Event,
    EventEnrichmentResult,
    Predicate,
    PredicateEnrichmentResult,
    RawAssertion,
)

from .queries import (
    CLEAR_SOURCE_ASSERTIONS,
    READ_SOURCE_ENTITIES,
    READ_SOURCE_EVENTS,
    READ_SOURCE_PREDICATES,
    REPLACE_SOURCE_ASSERTIONS,
    UPDATE_ENTITY_ENRICHMENT,
    UPDATE_EVENT_ENRICHMENT,
    UPDATE_PREDICATE_ENRICHMENT,
)


class SemanticRepository:
    """Access raw semantic occurrences and their typed enrichment fields."""

    def __init__(self, session_factory: Callable) -> None:
        self._session_factory = session_factory

    async def replace_source_assertions(
        self,
        source_uuid: str,
        assertions: list[RawAssertion],
    ) -> None:
        """Replace all raw semantic occurrences for one source."""
        if not assertions:
            async with self._session_factory() as session:
                result = await session.run(
                    CLEAR_SOURCE_ASSERTIONS,
                    source_uuid=source_uuid,
                )
                await result.consume()
            return

        parameters = {
            'source_uuid': source_uuid,
            'triplets': [
                {
                    'uuid': assertion.triplet.uuid,
                    'source_uuid': assertion.triplet.source_uuid,
                    'source_block_uuid': assertion.triplet.source_block_uuid,
                    'subject_uuid': assertion.triplet.subject_uuid,
                    'object_uuid': assertion.triplet.object_uuid,
                    'predicate_uuid': assertion.triplet.predicate_uuid,
                }
                for assertion in assertions
            ],
            'entities': [
                _endpoint_row(assertion.subject)
                for assertion in assertions
                if isinstance(assertion.subject, Entity)
            ]
            + [
                _endpoint_row(assertion.object)
                for assertion in assertions
                if isinstance(assertion.object, Entity)
            ],
            'events': [
                _endpoint_row(assertion.subject)
                for assertion in assertions
                if isinstance(assertion.subject, Event)
            ]
            + [
                _endpoint_row(assertion.object)
                for assertion in assertions
                if isinstance(assertion.object, Event)
            ],
            'predicates': [
                {
                    'uuid': assertion.predicate.uuid,
                    'source_uuid': assertion.predicate.source_uuid,
                    'source_block_uuid': assertion.predicate.source_block_uuid,
                    'predicate': assertion.predicate.predicate,
                }
                for assertion in assertions
            ],
        }
        async with self._session_factory() as session:
            result = await session.run(REPLACE_SOURCE_ASSERTIONS, **parameters)
            await result.consume()

    async def load_entities(self, source_uuid: str) -> list[Entity]:
        """Load raw entity occurrences for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITIES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            Entity(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
            )
            for row in rows
        ]

    async def load_events(self, source_uuid: str) -> list[Event]:
        """Load raw event occurrences for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_EVENTS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            Event(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
            )
            for row in rows
        ]

    async def load_predicates(self, source_uuid: str) -> list[Predicate]:
        """Load raw predicate occurrences for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PREDICATES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            Predicate(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                predicate=row['predicate'],
            )
            for row in rows
        ]

    async def update_entity_enrichment(
        self,
        source_uuid: str,
        results: list[EntityEnrichmentResult],
    ) -> None:
        """Update descriptions and embeddings on existing entity nodes."""
        await self._update_enrichment(
            UPDATE_ENTITY_ENRICHMENT,
            source_uuid,
            results,
        )

    async def update_event_enrichment(
        self,
        source_uuid: str,
        results: list[EventEnrichmentResult],
    ) -> None:
        """Update descriptions and embeddings on existing event nodes."""
        await self._update_enrichment(
            UPDATE_EVENT_ENRICHMENT,
            source_uuid,
            results,
        )

    async def update_predicate_enrichment(
        self,
        source_uuid: str,
        results: list[PredicateEnrichmentResult],
    ) -> None:
        """Update descriptions and embeddings on existing predicate nodes."""
        await self._update_enrichment(
            UPDATE_PREDICATE_ENRICHMENT,
            source_uuid,
            results,
        )

    async def _update_enrichment(
        self,
        query: str,
        source_uuid: str,
        results: list[
            EntityEnrichmentResult
            | EventEnrichmentResult
            | PredicateEnrichmentResult
        ],
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


def _endpoint_row(endpoint: Entity | Event) -> dict[str, str]:
    """Serialize one typed endpoint occurrence."""
    return {
        'uuid': endpoint.uuid,
        'source_uuid': endpoint.source_uuid,
        'source_block_uuid': endpoint.source_block_uuid,
        'name': endpoint.name,
    }
