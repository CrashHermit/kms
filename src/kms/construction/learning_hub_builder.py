import asyncio
from collections.abc import Callable

from kms import config
from kms.core import clustering, content, embeddings
from kms.graph import learning_hubs, queries, writer


async def build_source_hubs(
    kind: str,
    source: str,
    records: list[dict],
    *,
    adjudicator,
    synthesizer,
) -> dict:
    stage = getattr(config.get_settings().stages, f'{kind}_hubs')
    if not records:
        return {'hubs': [], 'records': 0}
    missing = [
        record['uuid'] for record in records if not record.get('embedding')
    ]
    if missing:
        raise RuntimeError(
            f'learning hub ({kind}): records lack embeddings: {missing[:5]}'
        )

    async def adjudicate(left: dict, right: dict) -> bool:
        return await adjudicator.aforward(
            left=_record_text(left),
            right=_record_text(right),
        )

    final_groups = await clustering.adjudicated_groups(
        records,
        recall_threshold=stage.recall_threshold,
        merge_above=stage.merge_above,
        separate_below=stage.separate_below,
        adjudicate=adjudicate,
        max_concurrency=stage.max_concurrent_calls,
    )
    return await _synthesize_groups(
        kind,
        final_groups,
        source=source,
        synthesizer=synthesizer,
        max_concurrency=stage.max_concurrent_calls,
    )


async def build_meta_hubs(
    kind: str,
    records: list[dict],
    *,
    adjudicator,
    synthesizer,
) -> dict:
    stage = getattr(config.get_settings().stages, f'{kind}_hubs')
    records = [record for record in records if record.get('source')]
    if not records:
        return {'hubs': [], 'records': 0}
    missing = [
        record['uuid'] for record in records if not record.get('embedding')
    ]
    if missing:
        raise RuntimeError(
            f'meta learning hub ({kind}): records lack embeddings: '
            f'{missing[:5]}'
        )

    async def adjudicate(left: dict, right: dict) -> bool:
        return await adjudicator.aforward(
            left=_record_text(left),
            right=_record_text(right),
        )

    final_groups = await clustering.adjudicated_groups(
        records,
        recall_threshold=stage.recall_threshold,
        merge_above=stage.merge_above,
        separate_below=stage.separate_below,
        adjudicate=adjudicate,
        max_concurrency=stage.max_concurrent_calls,
    )
    qualified = [
        group
        for group in final_groups
        if len({record['source'] for record in group}) >= 2
    ]
    result = await _synthesize_groups(
        kind,
        qualified,
        source=None,
        synthesizer=synthesizer,
        max_concurrency=stage.max_concurrent_calls,
    )
    return {'hubs': result['hubs'], 'records': len(records)}


def _record_text(record: dict) -> str:
    description = record.get('description') or ''
    if not description:
        raise RuntimeError(
            f'learning hub input {record["uuid"]} lacks a description'
        )
    return description


async def _synthesize_groups(
    kind: str,
    groups: list[list[dict]],
    *,
    source: str | None,
    synthesizer,
    max_concurrency: int,
) -> dict:
    gate = asyncio.Semaphore(max_concurrency)

    async def synthesize(group: list[dict]) -> dict:
        async with gate:
            canonical_name, description = await synthesizer.aforward(
                evidence=[_record_text(record) for record in group]
            )
        members = [record['uuid'] for record in group]
        hub = {
            'members': members,
            'canonical_name': canonical_name,
            'description': description,
        }
        if source is None:
            hub['sources'] = sorted({record['source'] for record in group})
        return hub

    synthesized = await asyncio.gather(*(synthesize(group) for group in groups))
    if not synthesized:
        return {'hubs': [], 'records': 0}
    vectors = await embeddings.embedder().embed(
        [
            content.Content.from_text(
                f'{hub["canonical_name"]}: {hub["description"]}'
            )
            for hub in synthesized
        ]
    )
    hubs = []
    for hub, vector in zip(synthesized, vectors, strict=True):
        hub['embedding'] = vector
        if source is None:
            hub['uuid'] = learning_hubs.meta_hub_uuid(kind, hub['members'])
        else:
            hub['uuid'] = learning_hubs.hub_uuid(kind, source, hub['members'])
            hub['source'] = source
        hubs.append(hub)
    return {'hubs': hubs, 'records': sum(len(group) for group in groups)}


async def rebuild_meta_hubs(
    kind: str,
    *,
    session_factory: Callable,
    adjudicator,
    synthesizer,
) -> dict:
    records = await queries.all_source_learning_hubs(
        session_factory,
        kind,
    )
    result = await build_meta_hubs(
        kind,
        records,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_meta_learning_hubs(
        kind,
        session_factory=session_factory,
    )
    await writer.persist_meta_learning_hubs(
        kind,
        result['hubs'],
        session_factory=session_factory,
    )
    return {
        'meta_hubs': len(result['hubs']),
        'source_hubs': result['records'],
    }
