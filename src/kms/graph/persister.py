"""
Pipeline stage that persists the structural node stream to Neo4j — the provenance layer.

Runs right after the splitter (the last stage to mutate and re-id the node stream) and before
the entity finders, so the ``:Source`` root and its ``:Node`` chain exist in the graph before any
entity work builds on top of them. The node ids it persists are the final ids the entity overlay's
``members`` reference, which is why it sits after the splitter, not after the seam merger.

Gated on configuration: if no Neo4j target is wired (``NEO4J_URI`` unset) or the run carries no
``source``, this is a no-op — DB-less runs (and the test suite) still complete, writing only the
JSON artifacts. The schema bootstrap is idempotent, so running it per book is safe.
"""

from kms.core.state import State
from kms.graph.db import is_configured
from kms.graph.schema import ensure_schema
from kms.graph.writer import persist_nodes


class NodePersisterNode:
    """Sequential stage: upsert the run's node stream as the graph's provenance layer."""

    async def run(self, state: State) -> dict:
        source = state.get("source")
        if not is_configured() or not source:
            return {}
        await ensure_schema()
        await persist_nodes(state.get("nodes", []), source, state.get("source_metadata"))
        return {}
