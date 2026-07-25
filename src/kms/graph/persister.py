"""
Pipeline stages that persist the graph to Neo4j: the structural node stream (provenance layer) and
the block overlay and its procedural layer on top of it.

``NodePersisterNode`` runs right after the splitter (the last stage to mutate and re-id the node
stream) and before the group finder, so the ``:Source`` root and its ``:Node`` chain exist in the
graph before any entity work builds on top of them. The node ids it persists are the final ids the
overlay's ``members`` reference, which is why it sits after the splitter, not after the seam
merger. ``EntityPersisterNode`` runs at the very end, once the statement extractor, the procedure
extractor and the instruction distributor have finished, so it sees the fully attributed overlay;
it orders the overlay into one document-ordered, globally-id'd list
(``models.flatten_entities``) and upserts the blocks and their procedures.

The concept layer is currently dark (``graph.concepts``) — its only source was the deleted ``field``
taxonomy — so nothing writes ``:Concept`` nodes or ``:INSTANCE_OF`` edges yet.

Both stages are gated on configuration: if no Neo4j target is wired (``NEO4J_URI`` unset) or the run
carries no ``source``, they are no-ops — DB-less runs (and the test suite) still complete end to
end, they just don't persist. The schema bootstrap is idempotent, so running it per book (and from
either stage) is safe.
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


class EntityPersisterNode:
    """Sequential terminal stage: order the block overlay and upsert it as the graph's ``:Entity``
    layer (rooted under the book's ``:Source``, linked to its member ``:Node`` s), then the
    procedural layer on top (``:Procedure`` per derivation, ``:Act`` per step).

    The procedures are written after the entities so the ``:HAS_PROCEDURE`` MATCH has vertices to
    attach to."""

    async def run(self, state: state.State) -> dict:
        """Order the block overlay and upsert the :Entity and procedural layers."""
        source = state.get('source')
        if not db.is_configured() or not source:
            return {}
        await schema.ensure_schema()
        entities = models.flatten_entities(
            state.get('entities', []), state.get('nodes', [])
        )
        await writer.persist_entities(entities, source)
        await writer.persist_procedures(entities, source)
        return {}
