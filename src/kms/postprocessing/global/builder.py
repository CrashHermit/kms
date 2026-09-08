import asyncio
from collections.abc import Awaitable, Sequence
from typing import Protocol

from kms import config
from kms.core import clustering, embeddings, models


class HubAdjudicator(Protocol):
    def aforward(self, *, left: str, right: str) -> Awaitable[bool]: ...


class HubSynthesizer(Protocol):
    def aforward(
        self, *, evidence: list[str]
    ) -> Awaitable[tuple[str, str]]: ...


async def build(
    records: Sequence[models.HubRecord],
    *,
    stage_name: str,
    adjudicator: HubAdjudicator,
    synthesizer: HubSynthesizer,
) -> dict:
    """Cluster persisted local hubs into global hubs."""
    if not records:
        return {'hubs': [], 'records': 0}
    stage = getattr(config.get_settings().stages, stage_name)
    missing = [record.uuid for record in records if not record.embedding]
    if missing:
        raise RuntimeError(
            f'{stage_name}: records lack embeddings: {missing[:5]}'
        )

    async def adjudicate(left: models.HubRecord, right: models.HubRecord):
        return await adjudicator.aforward(
            left=left.description or '', right=right.description or ''
        )

    groups = await clustering.adjudicated_groups(
        list(records),
        recall_threshold=stage.recall_threshold,
        merge_above=stage.merge_above,
        separate_below=stage.separate_below,
        adjudicate=adjudicate,
        max_concurrency=stage.max_concurrent_calls,
        comparison_token_budget=stage.comparison_token_budget,
    )
    groups = [
        group
        for group in groups
        if len({record.source for record in group}) >= 2
    ]
    gate = asyncio.Semaphore(stage.max_concurrent_calls)

    async def synthesize(group: Sequence[models.HubRecord]) -> dict:
        async with gate:
            name, description = await synthesizer.aforward(
                evidence=[record.description or '' for record in group]
            )
        return {
            'members': [record.uuid for record in group],
            'canonical_name': name,
            'description': description,
            'sources': sorted({record.source for record in group}),
        }

    synthesized = await asyncio.gather(*(synthesize(group) for group in groups))
    if not synthesized:
        return {'hubs': [], 'records': 0}
    vectors = await embeddings.embedder().embed(
        [
            f'{hub["canonical_name"]}: {hub["description"]}'
            for hub in synthesized
        ]
    )
    for hub, vector in zip(synthesized, vectors, strict=True):
        hub['embedding'] = vector
    return {'hubs': synthesized, 'records': sum(len(group) for group in groups)}
