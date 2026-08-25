"""Project completed construction state into Neo4j."""

from collections.abc import Callable

from kms.core import embeddings, models, state
from kms.graph import queries, schema, writer


class FinalProjectorNode:
    """Persists the complete construction result at one graph boundary."""

    def __init__(
        self,
        session_factory: Callable | None,
        neo4j_configured: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self._neo4j_configured = neo4j_configured

    async def run(self, current_state: dict) -> dict:
        """Projects all raw and derived construction artifacts."""
        bundle = state.to_construction_bundle(current_state)
        source = bundle.source.key
        if not self._neo4j_configured or not source:
            return {'projected': False}
        models.validate_bundle(bundle, require_identities=True)

        await schema.ensure_schema(self._session_factory)
        await self._persist_raw(bundle, source)
        await self._persist_hubs(bundle, source)
        await self._persist_derived(bundle, source)
        return {'projected': True}

    async def _persist_raw(self, bundle, source: str) -> None:
        nodes = bundle.nodes
        node_embeddings = await embeddings.embed_source_nodes(nodes)
        statements = bundle.statements
        procedures = bundle.procedures
        await writer.persist_nodes(
            nodes,
            source,
            session_factory=self._session_factory,
            metadata=bundle.source.metadata,
            embeddings=node_embeddings,
        )
        await writer.persist_statements(
            statements, nodes, source, session_factory=self._session_factory
        )
        await writer.persist_procedures(
            procedures, nodes, source, session_factory=self._session_factory
        )
        await writer.persist_statement_procedure_links(
            statements,
            procedures,
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_instructions(
            bundle.instructions,
            bundle.nodes,
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_assertions(
            bundle.triplets,
            source,
            bundle.nodes,
            session_factory=self._session_factory,
            entity_descriptions=bundle.entity_descriptions,
            predicate_descriptions=bundle.predicate_descriptions,
            entity_embeddings=bundle.entity_embeddings,
            predicate_embeddings=bundle.predicate_embeddings,
        )
        await writer.persist_chain(
            nodes, source, session_factory=self._session_factory
        )

    async def _persist_hubs(self, bundle, source: str) -> None:
        entity_records = bundle.entity_hub_records
        if entity_records:
            await writer.persist_entity_hubs(
                entity_records,
                session_factory=self._session_factory,
                subsumption_edges=[],
                tier='source',
            )
        if bundle.entity_hub_assignments:
            await writer.attach_entity_components(
                bundle.entity_hub_assignments,
                aliases=[],
                session_factory=self._session_factory,
            )

        predicate_records = bundle.predicate_hub_records
        if predicate_records:
            await writer.persist_predicate_hubs(
                predicate_records,
                session_factory=self._session_factory,
                subsumption_edges=[],
                tier='source',
            )
        if bundle.predicate_hub_assignments:
            await writer.attach_predicate_components(
                bundle.predicate_hub_assignments,
                aliases=[],
                session_factory=self._session_factory,
            )

        triplet_hubs = bundle.triplet_hubs
        if triplet_hubs:
            await writer.clear_triplet_hubs(
                'source', session_factory=self._session_factory, source=source
            )
            await writer.persist_triplet_hubs(
                triplet_hubs,
                tier='source',
                session_factory=self._session_factory,
            )

    async def _persist_derived(self, bundle, source: str) -> None:
        await writer.persist_statement_enrichment(
            bundle.statement_enrichments,
            session_factory=self._session_factory,
        )
        await writer.persist_procedure_enrichment(
            bundle.procedure_enrichments,
            session_factory=self._session_factory,
        )
        await writer.clear_statement_hubs(
            source, session_factory=self._session_factory
        )
        await writer.persist_statement_hubs(
            bundle.statement_hubs,
            session_factory=self._session_factory,
        )
        await writer.clear_procedure_hubs(
            source, session_factory=self._session_factory
        )
        await writer.persist_procedure_hubs(
            bundle.procedure_hubs,
            session_factory=self._session_factory,
        )
        links = bundle.procedure_links
        if links:
            async with self._session_factory() as session:
                await session.run(
                    queries.MERGE_HAS_PROCEDURE,
                    pairs=[
                        {
                            'statement': link.statement_uuid,
                            'procedure': link.procedure_uuid,
                        }
                        for link in links
                    ],
                    now=writer.utcnow_iso(),
                )
