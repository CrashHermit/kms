"""
Collector — the entity layer's fan-in: whichever finder overlays ran become one list.

Every stage upstream of here writes its own sparse overlay onto its own channel: three of them on
the per-type entity layer (problem / definition / theorem, running in parallel), one on the
block-finder layer. Every stage downstream — the conceptualizer, the dependency finder, the entity
persister — wants the opposite: a single document-ordered list with stable global ids.

This is that seam, and it exists as its own stage for two reasons. It makes the *number* of overlays
a wiring detail rather than something three separate consumers each have to know (which is what the
generalization needs: the block-finder layer collapses three channels into one — see
``docs/GENERALIZATION.md``). And it moves the flattening out of the graph persister, so the ids the
conceptualizer and the dependency finder work with are the same ids the graph is keyed on, whether or
not Neo4j is configured — a DB-less run collects and conceptualizes exactly as a persisting one does.

The flattening itself is ``core.models.flatten_entities``: concatenate, order by each entity's first
member's position in the node stream, and stamp the document-order ``id`` the graph tier derives its
vertex uuids from.
"""

from kms.core import models, state

# The overlay channels a finder may have written. Read defensively — only the layer that actually ran
# has a populated channel, and an unwired channel is simply absent from the state.
OVERLAY_CHANNELS = (
    'block_entities',
    'problem_entities',
    'definition_entities',
    'theorem_entities',
)


class CollectorNode:
    """Sequential fan-in stage: flatten the finder overlays into the ``entities`` channel."""

    async def run(self, state: state.State) -> dict:
        """Flattens every populated overlay channel into one document-ordered, globally-id'd list."""
        overlays = [
            state.get(channel, []) or [] for channel in OVERLAY_CHANNELS
        ]
        return {
            'entities': models.flatten_entities(
                overlays, state.get('nodes', [])
            )
        }
