"""Graph node that persists the whole extraction result to Neo4j."""

from collections.abc import Callable

from kms.core import state
from kms.graph import schema, writer


class IngestionPersisterNode:
    """Persists nodes, statements, procedures, and assertions in order.

    A no-op when Neo4j is not configured or the state has no source.
    """

    def __init__(
        self,
        session_factory: Callable | None = None,
        neo4j_configured: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self._neo4j_configured = neo4j_configured

    async def run(self, state: state.State) -> dict:
        """Runs the persistence writes for the current pipeline state.

        Args:
            state: The pipeline state holding the extracted knowledge.

        Returns:
            An empty update dict.
        """
        source = state.get('source')
        if not self._neo4j_configured or not source:
            return {}
        await schema.ensure_schema(self._session_factory)

        nodes = state.get('nodes', [])
        await writer.persist_nodes(
            nodes,
            source,
            session_factory=self._session_factory,
            metadata=state.get('source_metadata'),
        )
        statements = state.get('statements', [])
        procedures = state.get('procedures', [])
        await writer.persist_statements(
            statements,
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_procedures(
            procedures,
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_statement_procedure_links(
            statements,
            procedures,
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_instructions(
            state.get('instructions', []),
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_assertions(
            state.get('triplets', []),
            source,
            session_factory=self._session_factory,
            entity_descriptions=state.get('entity_descriptions', {}),
            predicate_descriptions=state.get('predicate_descriptions', {}),
            entity_embeddings=state.get('entity_embeddings', {}),
            predicate_embeddings=state.get('predicate_embeddings', {}),
        )
        await writer.persist_chain(
            nodes,
            source,
            session_factory=self._session_factory,
        )
        return {}
