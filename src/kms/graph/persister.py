"""
Pipeline stages that persist the graph to Neo4j: the structural node stream (provenance layer) and
the entity overlay on top of it.

``NodePersisterNode`` runs right after the splitter (the last stage to mutate and re-id the node
stream) and before the entity finders, so the ``:Source`` root and its ``:Node`` chain exist in the
graph before any entity work builds on top of them. The node ids it persists are the final ids the
entity overlay's ``members`` reference, which is why it sits after the splitter, not after the seam
merger. ``EntityPersisterNode`` runs at the very end, over the ``entities`` channel the collector
already flattened (and the conceptualizer and dependency finder have since enriched), so it sees the
finished overlay and does no assembly of its own — it upserts ``:Entity`` vertices linked back to the
``:Node`` chunks and then every semantic layer above them.

Both are gated on configuration: if no Neo4j target is wired (``NEO4J_URI`` unset) or the run carries
no ``source``, they are no-ops — DB-less runs (and the test suite) still complete end to end, they
just don't persist. The schema bootstrap is idempotent, so running it per book (and from either
stage) is safe.
"""

from kms.core import state
from kms.graph.db import is_configured
from kms.graph.schema import ensure_schema
from kms.graph.writer import (
    persist_concepts,
    persist_dependencies,
    persist_entities,
    persist_nodes,
    persist_procedures,
    persist_realizes,
    persist_references,
    persist_uses,
)


class NodePersisterNode:
    """Sequential stage: upsert the run's node stream as the graph's provenance layer."""

    async def run(self, state: state.State) -> dict:
        """Upsert the run's node stream as the graph's provenance layer."""
        source = state.get('source')
        if not is_configured() or not source:
            return {}
        await ensure_schema()
        await persist_nodes(
            state.get('nodes', []), source, state.get('source_metadata')
        )
        return {}


class EntityPersisterNode:
    """Terminal stage: upsert the finished overlay as the graph's ``:Entity`` layer (rooted under the
    book's ``:Source``, linked to their member ``:Node`` s), then every layer above it.

    Write order is a dependency order, since each layer attaches to vertices the previous one made:
    the procedural layer (``:Procedure`` / ``:Event``) hangs off the entities; the concept layer
    (``:Concept`` + ``:INSTANCE_OF``) attaches to both entities and procedure events; the concept
    ``:DEPENDS_ON`` prerequisites join concepts that must already exist; the cross-entity reference
    layer mints the ``:Entity:Canonical`` targets; the step-level ``:USES`` edges need both an
    ``:Event`` and a ``:Canonical``; and the ``:REALIZES`` identity edges tie each canonical back to
    the in-corpus mention that defines/states it, so they run last."""

    async def run(self, state: state.State) -> dict:
        """Upsert the entity overlay and the procedural, concept, dependency, reference, :USES and
        :REALIZES layers above it."""
        source = state.get('source')
        if not is_configured() or not source:
            return {}
        await ensure_schema()
        entities = state.get('entities', [])
        await persist_entities(entities, source)
        await persist_procedures(entities, source)
        await persist_concepts(entities, source)
        await persist_dependencies(state.get('concept_dependencies', []))
        await persist_references(entities, source)
        await persist_uses(entities, source)
        await persist_realizes(entities, source)
        return {}
