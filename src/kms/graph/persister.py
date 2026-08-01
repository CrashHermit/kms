"""
Pipeline stage that persists the full graph to Neo4j at the end of the run.

``IngestionPersisterNode`` runs after the equation/variable node, once the
``:Source``/``:Node`` provenance tier and the ``:Statement``/``:Procedure``
overlay are complete in memory. It writes everything in one pass.

The persister is gated on configuration: if no Neo4j target is wired
(``NEO4J_URI`` unset) or the run carries no ``source``, it is a no-op.
"""

from kms.core import state
from kms.graph import db, schema, writer


class IngestionPersisterNode:
    """Terminal stage: upserts the ``:Source``/``:Node`` provenance tier
    and the ``:Statement``/``:Procedure`` overlay."""

    async def run(self, state: state.State) -> dict:
        """Persist everything."""
        source = state.get('source')
        if not db.is_configured() or not source:
            return {}
        await schema.ensure_schema()

        nodes = state.get('nodes', [])
        await writer.persist_nodes(nodes, source, state.get('source_metadata'))

        # The three tiers arrive on separate channels: `nodes` is the verbatim
        # page, `statements` and `procedures` the hub overlays over it.
        # ``persist_chain`` writes the pure provenance spine;
        # ``persist_statements`` / ``persist_procedures`` point each hub at
        # its member nodes via ``:MEMBER_OF``.
        statements = state.get('statements', [])
        procedures = state.get('procedures', [])
        await writer.persist_statements(statements, source)
        await writer.persist_procedures(procedures, source)
        # Governance hubs: the lead-ins removed from the stream, each pointing
        # at the exercises it governs. Written after the nodes exist, since
        # :GOVERNS matches them by uuid.
        await writer.persist_instructions(
            state.get('instructions', []), source
        )
        await writer.persist_equations(
            state.get('equations', []), source
        )
        await writer.persist_variables(
            state.get('variables', []), source
        )
        await writer.persist_chain(nodes, source)
        return {}
