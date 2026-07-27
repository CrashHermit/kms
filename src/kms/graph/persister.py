"""
Pipeline stages that persist the graph to Neo4j: the structural node stream (provenance layer).

``NodePersisterNode`` runs after the splitter, instruction finder and instruction distributor
(the last stages to mutate the node stream) and before the group finder, so the ``:Source`` root
and its ``:Node`` chain exist in the graph before any entity work builds on top of them. The node
ids it persists are the final ids the overlay's ``members`` reference, which is why it sits after
the stream-mutating stages, not after the seam merger. Instruction nodes have already been removed
by the distributor, so they are never persisted.

The entity and procedural layers are currently dark — they will be rewritten to follow
AutoSchemaKG's triple-extraction approach.

The persister is gated on configuration: if no Neo4j target is wired (``NEO4J_URI`` unset) or the
run carries no ``source``, it is a no-op — DB-less runs (and the test suite) still complete end to
end, they just don't persist. The schema bootstrap is idempotent, so running it per book is safe.
"""

from kms.core import models, state
from kms.graph import db, schema, writer


class NodePersisterNode:
    """Sequential stage: upsert the run's node stream as the graph's provenance layer."""

    async def run(self, state: state.State) -> dict:
        """Upsert the run's node stream as the graph's provenance layer."""
        source = state.get('source')
        if not db.is_configured() or not source:
            return {}
        await schema.ensure_schema()
        await writer.persist_nodes(
            state.get('nodes', []), source, state.get('source_metadata')
        )
        return {}
