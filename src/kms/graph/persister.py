"""
Pipeline stage that persists the full graph to Neo4j at the end of the run.

``IngestionPersisterNode`` runs after the equation/variable node, once the
``:Source``/``:Node`` provenance tier and the ``:Statement``/``:Procedure``
overlay are complete in memory. It writes everything in one pass.

The persister receives its Neo4j session factory at construction time.
When ``neo4j_configured`` is False, ``run`` is a no-op.
"""

from collections.abc import Callable

from kms.core import state
from kms.graph import schema, writer


class IngestionPersisterNode:
    """Terminal stage: upserts the ``:Source``/``:Node`` provenance tier
    and the ``:Statement``/``:Procedure`` overlay.

    Args:
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        neo4j_configured: Whether a Neo4j target is wired. When False,
            ``run`` is a no-op.
    """

    def __init__(
        self,
        session_factory: Callable | None = None,
        neo4j_configured: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self._neo4j_configured = neo4j_configured

    async def run(self, state: state.State) -> dict:
        """Persist everything."""
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

        # The three tiers arrive on separate channels: `nodes` is the verbatim
        # page, `statements` and `procedures` the hub overlays over it.
        # ``persist_chain`` writes the pure provenance spine;
        # ``persist_statements`` / ``persist_procedures`` point each hub at
        # its member nodes via ``:MEMBER_OF``.
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
        # Governance hubs: the lead-ins removed from the stream, each
        # pointing at the exercises it governs. Written after the nodes
        # exist, since :GOVERNS matches them by uuid.
        await writer.persist_instructions(
            state.get('instructions', []),
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_equations(
            state.get('equations', []),
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_variables(
            state.get('variables', []),
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_facts(
            state.get('atomic_facts', []),
            source,
            session_factory=self._session_factory,
        )
        await writer.persist_chain(
            nodes,
            source,
            session_factory=self._session_factory,
        )
        return {}
