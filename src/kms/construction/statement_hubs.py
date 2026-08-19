import asyncio

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import clustering, content, embeddings, models, module
from kms.graph import queries, writer
from kms.graph import statement_hubs as graph_hubs


class StatementHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared statement.'
    )
    description: str = Field(
        description=(
            'A standalone learner-facing description of the shared statement.'
        )
    )


class StatementHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable source-local learning target from statements that
    express the same claim, fact, theorem, explanation, or question pattern.

    The supplied descriptions are already enriched source-level statements.
    Preserve the supported meaning, qualifiers, and mathematical notation.
    Generalize across the supplied statements only where they share meaning.
    Do not mention the source, passages, statement identifiers, or procedures.
    Do not solve a question or invent facts.
    """

    evidence: list[str] = dspy.InputField(
        description='Descriptions of statements assigned to one local hub.'
    )
    result: StatementHubResult = dspy.OutputField(
        description='A reusable name and learner-facing statement description.'
    )


class StatementHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two enriched statements express the same reusable learning
    target within one source.

    Return True only when they communicate the same claim, fact, theorem,
    explanation, or question pattern with equivalent meaning. Return False for
    merely related, sequential, broader, narrower, or differently solved
    statements. Ignore whether either statement has a procedure.
    """

    left: str = dspy.InputField(description='The first enriched statement.')
    right: str = dspy.InputField(description='The second enriched statement.')
    should_merge: bool = dspy.OutputField(
        description='Whether both statements belong to one local StatementHub.'
    )


class StatementHubSynthesizer(module.Module):
    signature = StatementHubSynthesisSignature
    record_name = 'statement_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'evidence': evidence}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return result.canonical_name, result.description


class StatementHubAdjudicator(module.Module):
    signature = StatementHubAdjudicationSignature
    record_name = 'statement_hub_adjudicator'

    def encode(self, left: str, right: str) -> dict:
        return {'left': left, 'right': right}

    def decode(self, prediction, **inputs) -> bool:
        return prediction.should_merge


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
            content.Content.from_text(
                f'{hub["canonical_name"]}: {hub["description"]}'
            )
            for hub in synthesized
        ]
    )
    hubs = []
    for hub, vector in zip(synthesized, vectors, strict=True):
        hub['embedding'] = vector
        hub['uuid'] = (
            graph_hubs.meta_hub_uuid(hub['members'])
            if meta
            else graph_hubs.hub_uuid(source, hub['members'])
        )
        if not meta:
            hub['source'] = source
        hubs.append(hub)
    return {'hubs': hubs, 'records': sum(len(group) for group in groups)}


class StatementHubNode:
    def __init__(
        self,
        adjudicator: StatementHubAdjudicator,
        synthesizer: StatementHubSynthesizer,
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
    adjudicator: StatementHubAdjudicator,
    synthesizer: StatementHubSynthesizer,
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


async def rebuild_meta(
    *,
    session_factory,
    adjudicator: StatementHubAdjudicator,
    synthesizer: StatementHubSynthesizer,
) -> dict:
    """Rebuild cross-source MetaStatementHub records."""
    rows = await queries.all_source_statement_hubs(session_factory)
    typed = _records(rows)
    result = await _build(
        '', typed, adjudicator=adjudicator, synthesizer=synthesizer, meta=True
    )
    await writer.clear_meta_statement_hubs(session_factory=session_factory)
    await writer.persist_meta_statement_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {'meta_hubs': len(result['hubs']), 'source_hubs': result['records']}
