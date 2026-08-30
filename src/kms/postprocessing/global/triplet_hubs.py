"""Global cross-source triplet-hub materialization."""

from collections.abc import Callable

import dspy

from kms.construction.triplet_hubs import _prepare_groups, _synthesize_groups
from kms.graph import queries, writer


async def rebuild_global(
    *,
    language_model: dspy.LM,
    session_factory: Callable,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuilds qualified cross-source GlobalTripletHub nodes."""
    rows = await queries.triplet_hub_groups(session_factory, 'meta')
    groups = _prepare_groups(rows, 'meta')
    groups = await _synthesize_groups(
        groups,
        language_model=language_model,
        max_concurrency=max_concurrency,
        tier='meta',
    )
    await writer.clear_triplet_hubs('meta', session_factory=session_factory)
    await writer.persist_triplet_hubs(
        groups,
        tier='meta',
        session_factory=session_factory,
    )
    return {
        'global_triplet_hubs': len(groups),
        'triplets': sum(len(group['triplets']) for group in groups),
    }
