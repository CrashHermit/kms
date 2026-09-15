"""Persistence repository for raw semantic assertion occurrences."""

from collections.abc import Callable

from kms2.core.model import (
    RawAssertion,
    SourceEntity,
    SourceEntityDescriptionResult,
    SourceEntitySimilarityMatch,
    SourceEvent,
    SourceEventDescriptionResult,
    SourceEventSimilarityMatch,
    SourcePredicate,
    SourcePredicateDescriptionResult,
    SourcePredicateSimilarityMatch,
)

from .queries import (
    CLEAR_SOURCE_ASSERTIONS,
    FIND_SIMILAR_SOURCE_ENTITIES,
    FIND_SIMILAR_SOURCE_EVENTS,
    FIND_SIMILAR_SOURCE_PREDICATES,
    READ_SOURCE_ENTITIES,
    READ_SOURCE_EVENTS,
    READ_SOURCE_PREDICATES,
    REPLACE_SOURCE_ASSERTIONS,
    UPDATE_SOURCE_ENTITY_DESCRIPTION,
    UPDATE_SOURCE_EVENT_DESCRIPTION,
    UPDATE_SOURCE_PREDICATE_DESCRIPTION,
)


class SemanticRepository:
    """Access raw semantic occurrences and their typed descriptions."""

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
            'source_entities': [
                _endpoint_row(assertion.subject)
                for assertion in assertions
                if isinstance(assertion.subject, SourceEntity)
            ]
            + [
                _endpoint_row(assertion.object)
                for assertion in assertions
                if isinstance(assertion.object, SourceEntity)
            ],
            'source_events': [
                _endpoint_row(assertion.subject)
                for assertion in assertions
                if isinstance(assertion.subject, SourceEvent)
            ]
            + [
                _endpoint_row(assertion.object)
                for assertion in assertions
                if isinstance(assertion.object, SourceEvent)
            ],
            'source_predicates': [
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

    async def load_source_entities(
        self, source_uuid: str
    ) -> list[SourceEntity]:
        """Load raw entity occurrences for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITIES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceEntity(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
            )
            for row in rows
        ]

    async def load_source_events(self, source_uuid: str) -> list[SourceEvent]:
        """Load raw event occurrences for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_EVENTS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceEvent(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
            )
            for row in rows
        ]

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

    async def update_source_entity_description(
        self,
        source_uuid: str,
        results: list[SourceEntityDescriptionResult],
    ) -> None:
        """Update descriptions and embeddings on existing entity nodes."""
        await self._update_description(
            UPDATE_SOURCE_ENTITY_DESCRIPTION,
            source_uuid,
            results,
        )

    async def update_source_event_description(
        self,
        source_uuid: str,
        results: list[SourceEventDescriptionResult],
    ) -> None:
        """Update descriptions and embeddings on existing event nodes."""
        await self._update_description(
            UPDATE_SOURCE_EVENT_DESCRIPTION,
            source_uuid,
            results,
        )

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

    async def find_similar_source_entities(
        self,
        source_entity_uuid: str,
        *,
        top_k: int,
    ) -> list[SourceEntitySimilarityMatch]:
        """Find nearest source entities using Neo4j's entity index."""
        rows = await self._find_similar(
            FIND_SIMILAR_SOURCE_ENTITIES,
            source_entity_uuid,
            top_k,
        )
        return [
            SourceEntitySimilarityMatch(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
                description=row['description'],
                score=row['score'],
            )
            for row in rows
        ]

    async def find_similar_source_events(
        self,
        source_event_uuid: str,
        *,
        top_k: int,
    ) -> list[SourceEventSimilarityMatch]:
        """Find nearest source events using Neo4j's event index."""
        rows = await self._find_similar(
            FIND_SIMILAR_SOURCE_EVENTS,
            source_event_uuid,
            top_k,
        )
        return [
            SourceEventSimilarityMatch(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                source_block_uuid=row['source_block_uuid'],
                name=row['name'],
                description=row['description'],
                score=row['score'],
            )
            for row in rows
        ]

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
        results: list[
            SourceEntityDescriptionResult
            | SourceEventDescriptionResult
            | SourcePredicateDescriptionResult
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


def _endpoint_row(endpoint: SourceEntity | SourceEvent) -> dict[str, str]:
    """Serialize one typed endpoint occurrence."""
    return {
        'uuid': endpoint.uuid,
        'source_uuid': endpoint.source_uuid,
        'source_block_uuid': endpoint.source_block_uuid,
        'name': endpoint.name,
    }
