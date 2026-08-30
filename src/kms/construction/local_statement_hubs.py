import asyncio

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import clustering, embeddings, models, module
from kms.graph import local_statement_hubs as graph_hubs
from kms.graph import queries, writer


class LocalStatementHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared statement.'
    )
    description: str = Field(
        description='A standalone canonical description of the shared statement.'
    )


class LocalStatementHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable source-local canonical statement from statements
    that express the same claim, fact, theorem, explanation, or question
    pattern. Preserve supported meaning and qualifiers. Do not solve or invent.
    """

    request: models.EvidenceInput = dspy.InputField()
    result: LocalStatementHubResult = dspy.OutputField()


class LocalStatementHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two enriched statements express the same canonical meaning
    within one source. Return True only for equivalent meaning, not merely
    related, sequential, broader, narrower, or differently solved statements.
    """

    pair: models.TextPairInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


class LocalStatementHubSynthesizer(module.Module):
    signature = LocalStatementHubSynthesisSignature
    record_name = 'statement_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'request': models.EvidenceInput(evidence=evidence)}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = LocalStatementHubResult.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


class LocalStatementHubAdjudicator(module.Module):
    signature = LocalStatementHubAdjudicationSignature
    record_name = 'statement_hub_adjudicator'

    def encode(self, left: str, right: str) -> dict:
        return {'pair': models.TextPairInput(left=left, right=right)}

    def decode(self, prediction, **inputs) -> bool:
        result = models.MergeDecision.model_validate(prediction.result)
        return module.require_bool(result.should_merge, 'should_merge')


def _records(rows: list[dict]) -> tuple[models.StatementHubRecord, ...]:
    return tuple(
        models.StatementHubRecord(
            uuid=row['uuid'],
            source=row['source'],
            description=row['description'],
            embedding=list(row['embedding']),
        )
        for row in rows
    )


async def _build(
    source: str,
    records,
    *,
    adjudicator,
    synthesizer,
    meta: bool = False,
) -> dict:
    stage = config.get_settings().stages.statement_hubs
    if not records:
        return {'hubs': [], 'records': 0}
    missing = [record.uuid for record in records if not record.embedding]
    if missing:
        raise RuntimeError(
            f'statement hubs: records lack embeddings: {missing[:5]}'
        )

    async def adjudicate(left, right):
        return await adjudicator.aforward(
            left=left.description,
            right=right.description,
        )

    groups = await clustering.adjudicated_groups(
        list(records),
        recall_threshold=stage.recall_threshold,
        merge_above=stage.merge_above,
        separate_below=stage.separate_below,
        adjudicate=adjudicate,
        max_concurrency=stage.max_concurrent_calls,
        comparison_token_budget=stage.comparison_token_budget,
        rerank_top_n=stage.rerank_top_n,
    )
    if meta:
        groups = [
            group for group in groups if len({r.source for r in group}) >= 2
        ]
    gate = asyncio.Semaphore(stage.max_concurrent_calls)

    async def synthesize(group):
        async with gate:
            name, description = await synthesizer.aforward(
                evidence=[record.description for record in group]
            )
        members = [record.uuid for record in group]
        return {
            'members': members,
            'canonical_name': name,
            'description': description,
            **({'sources': sorted({r.source for r in group})} if meta else {}),
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
    hubs = []
    for hub, vector in zip(synthesized, vectors, strict=True):
        hub['embedding'] = vector
        hub['uuid'] = (
            graph_hubs.global_hub_uuid(hub['members'])
            if meta
            else graph_hubs.hub_uuid(source, hub['members'])
        )
        if not meta:
            hub['source'] = source
        hubs.append(hub)
    return {'hubs': hubs, 'records': sum(len(group) for group in groups)}


class LocalStatementHubNode:
    def __init__(
        self,
        adjudicator: LocalStatementHubAdjudicator,
        synthesizer: LocalStatementHubSynthesizer,
    ) -> None:
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        source = bundle.source.key
        if not source:
            return {'construction_bundle': bundle}
        records = tuple(bundle.statement_hub_records)
        result = await _build(
            source,
            records,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        bundle.statement_hubs = result['hubs']
        return {
            'statement_hubs_created': len(result['hubs']),
            'statements_clustered': result['records'],
            'statement_hubs': bundle.statement_hubs,
            'construction_bundle': bundle,
        }


async def rebuild(
    source: str,
    *,
    session_factory,
    adjudicator: LocalStatementHubAdjudicator,
    synthesizer: LocalStatementHubSynthesizer,
) -> dict:
    records = _records(
        await queries.statement_hub_items(session_factory, source)
    )
    result = await _build(
        source, records, adjudicator=adjudicator, synthesizer=synthesizer
    )
    await writer.clear_statement_hubs(source, session_factory=session_factory)
    await writer.persist_statement_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {
        'statement_hubs': len(result['hubs']),
        'statements': result['records'],
    }
