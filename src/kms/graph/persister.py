"""
Pipeline stage that persists the full graph to Neo4j at the end of the run.

``IngestionPersisterNode`` runs after the procedure extractor, once the
``:Source``/``:Node`` provenance tier and the ``:Statement``/``:Procedure``
overlay are complete in memory. It writes everything in one pass.

The persister is gated on configuration: if no Neo4j target is wired
(``NEO4J_URI`` unset) or the run carries no ``source``, it is a no-op.
"""

from kms.core import models, state
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

        # Build ordered statement list from state.
        statement_ids = state.get('statement_ids', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        statements: list[models.StatementNode] = []
        for statement_id in statement_ids:
            node = nodes_by_id.get(statement_id)
            if isinstance(node, models.StatementNode):
                statements.append(node)

        await writer.persist_statements(statements, source)
        await writer.persist_procedures(statements, source)
        await writer.persist_chain(nodes, statements, source)
        return {}
