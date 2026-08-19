import asyncio

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import clustering, content, embeddings, models, module
from kms.graph import procedure_hubs as graph_hubs
from kms.graph import queries, writer


class ProcedureHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared method.'
    )
    description: str = Field(
        description=(
            'A standalone learner-facing description of the shared method.'
        )
    )


class ProcedureHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable source-local learning method from procedures that
    use the same general reasoning or action pattern.

    The supplied descriptions are enriched procedures. Explain the shared
    goal and method without copying source-specific names, values, answers, or
    navigation. Preserve supported conditions and mathematical notation. Do
    not invent steps or collapse merely related but different methods.
    """

    evidence: list[str] = dspy.InputField(
        description='Descriptions of procedures assigned to one local hub.'
    )
    result: ProcedureHubResult = dspy.OutputField(
        description='A reusable name and learner-facing method description.'
    )


class ProcedureHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two enriched procedures use the same reusable method within
    one source.

    Return True only when their general reasoning or action pattern is the
    same. Return False when they merely concern the same subject, share a
    result, or use materially different methods.
    """

    left: str = dspy.InputField(description='The first enriched procedure.')
    right: str = dspy.InputField(description='The second enriched procedure.')
    should_merge: bool = dspy.OutputField(
        description='Whether both procedures belong to one local ProcedureHub.'
    )


class ProcedureHubSynthesizer(module.Module):
    signature = ProcedureHubSynthesisSignature
    record_name = 'procedure_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'evidence': evidence}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return result.canonical_name, result.description


class ProcedureHubAdjudicator(module.Module):
    signature = ProcedureHubAdjudicationSignature
    record_name = 'procedure_hub_adjudicator'

    def encode(self, left: str, right: str) -> dict:
        return {'left': left, 'right': right}

    def decode(self, prediction, **inputs) -> bool:
        return prediction.should_merge


def _records(rows: list[dict]) -> tuple[models.ProcedureHubRecord, ...]:
    return tuple(
        models.ProcedureHubRecord(
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
    stage = config.get_settings().stages.procedure_hubs
    if not records:
        return {'hubs': [], 'records': 0}
    missing = [record.uuid for record in records if not record.embedding]
    if missing:
        raise RuntimeError(
            f'procedure hubs: records lack embeddings: {missing[:5]}'
        )

    async def adjudicate(left, right):
        return await adjudicator.aforward(
            left=left.description, right=right.description
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


class ProcedureHubNode:
    def __init__(
        self,
        adjudicator: ProcedureHubAdjudicator,
        synthesizer: ProcedureHubSynthesizer,
    ) -> None:
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        source = bundle.source.key
        if not source:
            return {'construction_bundle': bundle}
        records = tuple(bundle.procedure_hub_records)
        result = await _build(
            source,
            records,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        bundle.procedure_hubs = result['hubs']
        return {
            'procedure_hubs_created': len(result['hubs']),
            'procedures_clustered': result['records'],
            'procedure_hubs': bundle.procedure_hubs,
            'construction_bundle': bundle,
        }


async def rebuild(
    source: str,
    *,
    session_factory,
    adjudicator: ProcedureHubAdjudicator,
    synthesizer: ProcedureHubSynthesizer,
) -> dict:
    records = _records(
        await queries.procedure_hub_items(session_factory, source)
    )
    result = await _build(
        source, records, adjudicator=adjudicator, synthesizer=synthesizer
    )
    await writer.clear_procedure_hubs(source, session_factory=session_factory)
    await writer.persist_procedure_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {
        'procedure_hubs': len(result['hubs']),
        'procedures': result['records'],
    }


async def rebuild_meta(
    *,
    session_factory,
    adjudicator: ProcedureHubAdjudicator,
    synthesizer: ProcedureHubSynthesizer,
) -> dict:
    """Rebuild cross-source MetaProcedureHub records."""
    rows = await queries.all_source_procedure_hubs(session_factory)
    typed = _records(rows)
    result = await _build(
        '', typed, adjudicator=adjudicator, synthesizer=synthesizer, meta=True
    )
    await writer.clear_meta_procedure_hubs(session_factory=session_factory)
    await writer.persist_meta_procedure_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {'meta_hubs': len(result['hubs']), 'source_hubs': result['records']}
