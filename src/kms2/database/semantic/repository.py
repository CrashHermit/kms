"""Persistence repository for raw semantic assertion occurrences."""

from collections.abc import Callable

from kms2.core.model import (
    RawAssertion,
    SourceEntity,
    SourceEntityDescriptionResult,
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubMember,
    SourceEntitySimilarityMatch,
    SourceEvent,
    SourceEventDescriptionResult,
    SourceEventHub,
    SourceEventHubCandidate,
    SourceEventHubMember,
    SourceEventSimilarityMatch,
    SourcePredicate,
    SourcePredicateDescriptionResult,
    SourcePredicateHub,
    SourcePredicateHubCandidate,
    SourcePredicateHubMember,
    SourcePredicateSimilarityMatch,
    SourceProcedure,
    SourceProcedureDescriptionResult,
    SourceProcedureHub,
    SourceProcedureHubCandidate,
    SourceProcedureHubMember,
    SourceStatement,
    SourceStatementDescriptionResult,
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubMember,
)

from .queries import (
    CLEAR_SOURCE_ASSERTIONS,
    DETECT_SOURCE_ENTITY_COMMUNITIES,
    DETECT_SOURCE_EVENT_COMMUNITIES,
    DETECT_SOURCE_PREDICATE_COMMUNITIES,
    DETECT_SOURCE_PROCEDURE_COMMUNITIES,
    DETECT_SOURCE_STATEMENT_COMMUNITIES,
    DROP_SOURCE_ENTITY_HUB_GRAPH,
    DROP_SOURCE_EVENT_HUB_GRAPH,
    DROP_SOURCE_PREDICATE_HUB_GRAPH,
    DROP_SOURCE_PROCEDURE_HUB_GRAPH,
    DROP_SOURCE_STATEMENT_HUB_GRAPH,
    FIND_SIMILAR_SOURCE_ENTITIES,
    FIND_SIMILAR_SOURCE_EVENTS,
    FIND_SIMILAR_SOURCE_PREDICATES,
    READ_SOURCE_ENTITIES,
    READ_SOURCE_ENTITY_HUB_CANDIDATES,
    READ_SOURCE_EVENT_HUB_CANDIDATES,
    READ_SOURCE_EVENTS,
    READ_SOURCE_PREDICATE_HUB_CANDIDATES,
    READ_SOURCE_PREDICATES,
    READ_SOURCE_PROCEDURE_HUB_CANDIDATES,
    READ_SOURCE_PROCEDURES,
    READ_SOURCE_STATEMENT_HUB_CANDIDATES,
    READ_SOURCE_STATEMENTS,
    REPLACE_SOURCE_ASSERTIONS,
    REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES,
    REPLACE_SOURCE_ENTITY_HUBS,
    REPLACE_SOURCE_EVENT_ACCEPTED_EDGES,
    REPLACE_SOURCE_EVENT_HUBS,
    REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES,
    REPLACE_SOURCE_PREDICATE_HUBS,
    REPLACE_SOURCE_PROCEDURE_ACCEPTED_EDGES,
    REPLACE_SOURCE_PROCEDURE_HUBS,
    REPLACE_SOURCE_STATEMENT_ACCEPTED_EDGES,
    REPLACE_SOURCE_STATEMENT_HUBS,
    UPDATE_SOURCE_ENTITY_DESCRIPTION,
    UPDATE_SOURCE_EVENT_DESCRIPTION,
    UPDATE_SOURCE_PREDICATE_DESCRIPTION,
    UPDATE_SOURCE_PROCEDURE_DESCRIPTION,
    UPDATE_SOURCE_STATEMENT_DESCRIPTION,
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

    async def load_source_procedures(
        self, source_uuid: str
    ) -> list[SourceProcedure]:
        """Load source-scoped procedure pointers for semantic description."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PROCEDURES,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceProcedure(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                member_block_uuids=row['member_block_uuids'],
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

    async def update_source_procedure_description(
        self,
        source_uuid: str,
        results: list[SourceProcedureDescriptionResult],
    ) -> None:
        """Update descriptions and embeddings on source procedure nodes."""
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
                UPDATE_SOURCE_PROCEDURE_DESCRIPTION,
                source_uuid=source_uuid,
                rows=rows,
            )
            await response.consume()

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

    async def read_source_entity_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourceEntityHubCandidate]:
        """Read canonical same-source entity pairs for filtering."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_ENTITY_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [SourceEntityHubCandidate.model_validate(row) for row in rows]

    async def read_source_event_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourceEventHubCandidate]:
        """Read canonical same-source event pairs for filtering."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_EVENT_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [SourceEventHubCandidate.model_validate(row) for row in rows]

    async def read_source_predicate_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourcePredicateHubCandidate]:
        """Read canonical same-source predicate pairs for filtering."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PREDICATE_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [SourcePredicateHubCandidate.model_validate(row) for row in rows]

    async def replace_source_entity_accepted_edges(
        self,
        source_uuid: str,
        pairs: list[SourceEntityHubCandidate],
    ) -> None:
        """Replace accepted entity similarity edges for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_ENTITY_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def replace_source_event_accepted_edges(
        self,
        source_uuid: str,
        pairs: list[SourceEventHubCandidate],
    ) -> None:
        """Replace accepted event similarity edges for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_EVENT_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def replace_source_predicate_accepted_edges(
        self,
        source_uuid: str,
        pairs: list[SourcePredicateHubCandidate],
    ) -> None:
        """Replace accepted predicate similarity edges for one source."""
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PREDICATE_ACCEPTED_EDGES,
                source_uuid=source_uuid,
                pairs=[pair.model_dump() for pair in pairs],
            )
            await result.consume()

    async def detect_source_entity_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourceEntityHubMember]]:
        """Detect overlapping communities among source entity occurrences."""
        graph_name = f'kms2-source-entity-hubs-{source_uuid}'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_SOURCE_ENTITY_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_SOURCE_ENTITY_COMMUNITIES,
                    source_uuid=source_uuid,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[SourceEntityHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        SourceEntityHubMember(
                            uuid=row['uuid'],
                            name=row['name'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_SOURCE_ENTITY_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def detect_source_event_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourceEventHubMember]]:
        """Detect overlapping communities among source event occurrences."""
        graph_name = f'kms2-source-event-hubs-{source_uuid}'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_SOURCE_EVENT_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_SOURCE_EVENT_COMMUNITIES,
                    source_uuid=source_uuid,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[SourceEventHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        SourceEventHubMember(
                            uuid=row['uuid'],
                            name=row['name'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_SOURCE_EVENT_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def detect_source_predicate_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourcePredicateHubMember]]:
        """Detect overlapping communities among source predicate occurrences."""
        graph_name = f'kms2-source-predicate-hubs-{source_uuid}'
        async with self._session_factory() as session:
            await (
                await session.run(
                    DROP_SOURCE_PREDICATE_HUB_GRAPH,
                    graph_name=graph_name,
                )
            ).consume()
            try:
                result = await session.run(
                    DETECT_SOURCE_PREDICATE_COMMUNITIES,
                    source_uuid=source_uuid,
                    graph_name=graph_name,
                    max_iterations=max_iterations,
                    min_association_strength=min_association_strength,
                )
                rows = await result.data()
                communities: dict[object, list[SourcePredicateHubMember]] = {}
                for row in rows:
                    communities.setdefault(row['community_id'], []).append(
                        SourcePredicateHubMember(
                            uuid=row['uuid'],
                            predicate=row['predicate'],
                            description=row['description'],
                        )
                    )
            finally:
                await (
                    await session.run(
                        DROP_SOURCE_PREDICATE_HUB_GRAPH,
                        graph_name=graph_name,
                    )
                ).consume()
        return [
            sorted(members, key=lambda member: member.uuid)
            for members in communities.values()
            if len(members) >= minimum_community_size
        ]

    async def replace_source_entity_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceEntityHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace source-local entity hubs and memberships."""
        rows = [
            {
                **hub.model_dump(),
                'member_uuids': member_uuids,
            }
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_ENTITY_HUBS,
                source_uuid=source_uuid,
                hubs=rows,
            )
            await result.consume()

    async def replace_source_event_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceEventHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace source-local event hubs and memberships."""
        rows = [
            {
                **hub.model_dump(),
                'member_uuids': member_uuids,
            }
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_EVENT_HUBS,
                source_uuid=source_uuid,
                hubs=rows,
            )
            await result.consume()

    async def replace_source_predicate_hubs(
        self,
        source_uuid: str,
        hubs: list[SourcePredicateHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace source-local predicate hubs and memberships."""
        rows = [
            {
                **hub.model_dump(),
                'member_uuids': member_uuids,
            }
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PREDICATE_HUBS,
                source_uuid=source_uuid,
                hubs=rows,
            )
            await result.consume()

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

    async def detect_source_procedure_communities(
        self,
        source_uuid: str,
        *,
        max_iterations: int,
        min_association_strength: float,
        minimum_community_size: int,
    ) -> list[list[SourceProcedureHubMember]]:
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

    async def replace_source_statement_hubs(
        self,
        source_uuid: str,
        hubs: list[SourceStatementHub],
        memberships: list[list[str]],
    ) -> None:
        await self._replace_hubs(
            REPLACE_SOURCE_STATEMENT_HUBS, source_uuid, hubs, memberships
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
