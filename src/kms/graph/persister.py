"""
Pipeline stages that persist the graph to Neo4j: the structural node stream (provenance layer) and
the entity overlay on top of it.

``NodePersisterNode`` runs right after the splitter (the last stage to mutate and re-id the node
stream) and before the entity finders, so the ``:Source`` root and its ``:Node`` chain exist in the
graph before any entity work builds on top of them. The node ids it persists are the final ids the
entity overlay's ``members`` reference, which is why it sits after the splitter, not after the seam
merger. ``EntityPersisterNode`` runs at the very end, once all three per-type chains (and the problem
chain's instruction distributor) have finished, so it sees the fully attributed overlay; it flattens
the three finder channels into one document-ordered, globally-id'd list (``flatten_entities``) and
upserts them as ``:Entity`` vertices linked back to the ``:Node`` chunks.

Both are gated on configuration: if no Neo4j target is wired (``NEO4J_URI`` unset) or the run carries
no ``source``, they are no-ops — DB-less runs (and the test suite) still complete end to end, they
just don't persist. The schema bootstrap is idempotent, so running it per book (and from either
stage) is safe.
"""

from kms.core.models import flatten_entities
from kms.core.state import State
from kms.graph.db import is_configured
from kms.graph.schema import ensure_schema
from kms.graph.writer import (
    persist_entities,
    persist_nodes,
    persist_procedures,
    persist_references,
)


class NodePersisterNode:
    """Sequential stage: upsert the run's node stream as the graph's provenance layer."""

    async def run(self, state: State) -> dict:
        source = state.get("source")
        if not is_configured() or not source:
            return {}
        await ensure_schema()
        await persist_nodes(state.get("nodes", []), source, state.get("source_metadata"))
        return {}


class EntityPersisterNode:
    """Sequential fan-in stage: flatten the three per-type overlays and upsert them as the graph's
    ``:Entity`` layer (rooted under the book's ``:Source``, linked to their member ``:Node`` s), then
    the procedural layer (``:Procedure`` / ``:Event`` for proofs and solutions), then the cross-entity
    reference layer (``:REFERENCES`` edges onto ``:Entity:Canonical`` targets). Procedures and references
    are written after the entities so the citing vertices exist to attach to."""

    async def run(self, state: State) -> dict:
        source = state.get("source")
        if not is_configured() or not source:
            return {}
        await ensure_schema()
        entities = flatten_entities(
            state.get("problem_entities", []),
            state.get("definition_entities", []),
            state.get("theorem_entities", []),
            state.get("nodes", []),
        )
        await persist_entities(entities, source)
        await persist_procedures(entities, source)
        await persist_references(entities, source)
        return {}
