"""
Pipeline stage that persists the full graph to Neo4j at the end of the run.

``IngestionPersisterNode`` runs after the procedure extractor, once the
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

        # The two tiers arrive on two channels: `nodes` is the verbatim page,
        # `statements` the overlay over it. ``persist_chain`` is what relates
        # them, slotting each statement in at its members' place.
        statements = state.get('statements', [])
        await writer.persist_statements(statements, source)
        await writer.persist_procedures(statements, source)
        await writer.persist_chain(nodes, statements, source)
        return {}
