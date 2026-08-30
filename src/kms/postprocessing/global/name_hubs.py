"""Global cross-source lexical name-hub materialization."""

import asyncio
from collections.abc import Callable

import dspy

from kms.construction.name_hubs import (
    _judged_groups,
    _LexicalDefinitionSynthesizer,
)
from kms.core import llm
from kms.graph import names, queries, writer


async def rebuild_global(
    kind: str,
    *,
    language_model: dspy.LM,
    session_factory: Callable,
    similarity_threshold: float = 0.9,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuilds qualified meta lexical hubs from source-local name hubs."""
    source_hubs = await queries.all_name_hubs(session_factory, kind)
    await writer.clear_global_name_hubs(
        kind,
        session_factory=session_factory,
    )
    if not source_hubs:
        return {'global_name_hubs': 0, 'local_name_hubs': 0}

    sources = {row['source'] for row in source_hubs if row.get('source')}
    if len(sources) < 2:
        raise RuntimeError(
            f'name hubs (meta/{kind}): requires at least two distinct '
            f'sources, found {len(sources)}'
        )

    gate = llm.gate(max_concurrency)
    groups = await _judged_groups(
        source_hubs,
        kind,
        language_model=language_model,
        similarity_threshold=similarity_threshold,
        gate=gate,
    )

    qualified_groups = [
        group
        for group in groups
        if len({row['source'] for row in group if row.get('source')}) >= 2
    ]
    synthesizer = _LexicalDefinitionSynthesizer(language_model)

    async def _synthesize(group: list[dict]) -> dict:
        surface_forms = list(dict.fromkeys(row['text'] for row in group))
        async with gate:
            canonical_index = await synthesizer.aforward(
                surface_forms=surface_forms,
                kind=kind,
            )
        if not 0 <= canonical_index < len(surface_forms):
            raise RuntimeError(
                'meta lexical name synthesizer returned an invalid '
                f'surface-form index: {canonical_index}'
            )
        canonical_form = surface_forms[canonical_index]
        aliases = sorted(
            {alias for row in group for alias in row.get('aliases', []) or []}
            | set(surface_forms)
        )
        member_ids = [row['uuid'] for row in group]
        return {
            'uuid': names.global_name_hub_uuid(kind, member_ids),
            'canonical_form': canonical_form,
            'aliases': aliases,
            'members': member_ids,
            'sources': sorted({row['source'] for row in group}),
        }

    hubs = await asyncio.gather(
        *(_synthesize(group) for group in qualified_groups)
    )
    await writer.persist_global_name_hubs(
        kind,
        list(hubs),
        session_factory=session_factory,
    )
    return {
        'global_name_hubs': len(hubs),
        'local_name_hubs': len(source_hubs),
    }
