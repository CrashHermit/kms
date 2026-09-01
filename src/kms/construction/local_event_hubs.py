"""Build source-local EventHub concepts from event components."""

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import (
    batching,
    context_window,
    embeddings,
    llm,
    models,
    module,
    reranker,
)
from kms.graph import hubs, queries, writer
from kms.graph import local_event_hubs as graph_event_hubs

logger = logging.getLogger(__name__)


@dataclass
class EventHubSpec:
    """Fixed event-hub metadata consumed by the builder."""

    graph: Any
    stage_name: str
    hub_id_factory: callable
    source_resolver: callable
    synthesis_context: str
    all_components: callable
    clear_hubs: callable
    persist_hubs: callable


class EventHubDefinition(BaseModel):
    """Canonical event concept synthesized from event evidence."""

    canonical_name: str = Field(
        description='The canonical name for the event concept.'
    )
    description: str = Field(
        description='A standalone canonical event description supported by the supplied evidence.'
    )


class EventHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one canonical event concept from source-grounded event
    mentions that have already been grouped as equivalent.

    Name the occurrence or event itself, not its consequence, motivation,
    property, result, or surrounding explanatory clause. Preserve the
    event's participants and role meaning in the description. Do not merge
    a battle with its victory, an appointment with the person appointed, or
    an event with a later consequence. Generalize only the common event
    supported by every supplied mention. Never invent time, participants,
    outcomes, or causal claims.
    """

    request: models.HubSynthesisInput = dspy.InputField()
    result: EventHubDefinition = dspy.OutputField()


class EventHubSynthesizer(module.Module):
    """Event-tailored canonical event synthesis."""

    signature = EventHubSynthesisSignature
    record_name = 'event_hub_synthesizer'

    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        module.Module.__init__(
            self,
            language_model or llm.module_lm(self.record_name),
            recorder=recorder,
        )

    def encode(
        self, surface_forms: list[str], descriptions: list[str], scope: str
    ) -> dict:
        return {
            'request': models.HubSynthesisInput(
                surface_forms=surface_forms,
                descriptions=descriptions,
                scope=scope,
            )
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = EventHubDefinition.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


def _source_hub_diagnostics() -> dict[str, int]:
    """Returns the cumulative counters for one source-local hub build."""
    return {
        key: 0
        for key in (
            'records',
            'remaining_records',
            'embedding_comparisons',
            'ambiguous_candidates',
            'reranker_requests',
            'reranker_candidates',
            'reranker_selected',
            'locked_groups',
            'synthesizer_calls',
            'final_hubs',
        )
    }


def _log_source_hub_diagnostics(
    diagnostics: dict[str, int], *, group: int | None = None
) -> None:
    details: dict[str, int] = dict(diagnostics)
    if group is not None:
        details['group'] = group
    logger.info('event hubs: %s', details)


def _mention_text(record: dict) -> str:
    """Renders a hub mention record as compact text for the reranker."""
    mention = models.HubMentionInput.from_record(record)
    parts = [mention.name]
    parts.extend(mention.aliases)
    if mention.description:
        parts.append(mention.description)
    return ' '.join(parts)


async def _rerank_ambiguous_candidates(
    pivot: dict,
    candidates: list[dict],
    *,
    comparison_token_budget: int,
    rerank_top_n: int,
    diagnostics: dict[str, int],
) -> list[dict]:
    """Returns reranker-selected candidates in token-bounded batches."""
    if not candidates or not reranker.is_configured():
        return []
    pivot_text = _mention_text(pivot)
    pivot_tokens = context_window.estimate_text_tokens(pivot_text)
    if pivot_tokens >= comparison_token_budget:
        raise ValueError(
            f'event hubs: pivot {pivot["uuid"]!r} requires {pivot_tokens} '
            f'estimated tokens; reranker budget is {comparison_token_budget}'
        )
    candidate_texts = [
        (candidate, _mention_text(candidate)) for candidate in candidates
    ]
    for candidate, text in candidate_texts:
        token_cost = context_window.estimate_text_tokens(text)
        if pivot_tokens + token_cost > comparison_token_budget:
            raise ValueError(
                f'event hubs: candidate {candidate["uuid"]!r} with pivot '
                f'{pivot["uuid"]!r} requires {pivot_tokens + token_cost} '
                f'estimated tokens; reranker budget is {comparison_token_budget}'
            )
    selected: list[dict] = []
    selected_uuids: set[str] = set()
    client = reranker.reranker()
    for batch in batching.token_batches(
        candidate_texts,
        token_cost=lambda item: context_window.estimate_text_tokens(item[1]),
        token_budget=comparison_token_budget - pivot_tokens,
    ):
        documents = [text for _, text in batch]
        diagnostics['reranker_requests'] += 1
        diagnostics['reranker_candidates'] += len(documents)
        results = await client.rerank(pivot_text, documents, top_n=rerank_top_n)
        for result in results:
            index = result['index']
            if 0 <= index < len(batch):
                diagnostics['reranker_selected'] += 1
                candidate = batch[index][0]
                if candidate['uuid'] not in selected_uuids:
                    selected.append(candidate)
                    selected_uuids.add(candidate['uuid'])
    return selected


async def _synthesize_definitions(
    clusters: list[list[dict]],
    synthesizer: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> list[dict]:
    """Synthesizes one hub definition per cluster, concurrently."""

    async def _one(cluster: list[dict]) -> dict:
        surface_forms = list(
            dict.fromkeys(
                form
                for member in cluster
                for form in [member['name'], *member.get('aliases', [])]
                if form
            )
        )
        descriptions = sorted(
            {
                member['description']
                for member in cluster
                if member.get('description')
            }
        )
        async with gate:
            canonical_name, description = await synthesizer.aforward(
                surface_forms=surface_forms,
                descriptions=descriptions,
                scope=scope,
            )
        return {'canonical_name': canonical_name, 'description': description}

    return list(await asyncio.gather(*(_one(cluster) for cluster in clusters)))


def _dominant_source(records: list[dict]) -> str:
    counts = Counter(
        record['source'] for record in records if record.get('source')
    )
    return counts.most_common(1)[0][0] if counts else 'unknown'


def source_hub_id_factory(graph: Any) -> callable:
    """Creates a deterministic source-hub id function for event hubs."""

    def make_id(records: list[dict], definition: dict) -> str:
        identity = '|'.join(sorted(record['uuid'] for record in records))
        return graph.hub_uuid(_dominant_source(records), identity)

    return make_id


EVENT_SPEC = EventHubSpec(
    graph=None,
    stage_name='event_hubs',
    hub_id_factory=lambda records, definition: graph_event_hubs.hub_uuid(
        _dominant_source(records),
        '|'.join(sorted(record['uuid'] for record in records)),
    ),
    source_resolver=_dominant_source,
    synthesis_context='Synthesize a source-local semantic event hub.',
    all_components=queries.all_event_components,
    clear_hubs=writer.clear_event_hubs,
    persist_hubs=writer.persist_event_hubs,
)


def _workflow_spec() -> EventHubSpec:
    """Returns graph-specific metadata for workflow event hubs."""
    return EventHubSpec(
        graph=hubs,
        stage_name=EVENT_SPEC.stage_name,
        hub_id_factory=lambda records, definition: hubs.hub_uuid(
            hubs.LOCAL_EVENT_HUB_LABEL,
            _dominant_source(records),
            '|'.join(sorted(record['uuid'] for record in records)),
        ),
        source_resolver=EVENT_SPEC.source_resolver,
        synthesis_context=EVENT_SPEC.synthesis_context,
        all_components=EVENT_SPEC.all_components,
        clear_hubs=EVENT_SPEC.clear_hubs,
        persist_hubs=EVENT_SPEC.persist_hubs,
    )


async def build_event_hubs(
    records: list[dict],
    *,
    source: str | None,
    spec: EventHubSpec,
    max_concurrency: int | None = None,
    synthesizer: Any,
) -> dict:
    """Clusters all source-local event components into locked hub groups."""
    diagnostics = _source_hub_diagnostics()
    diagnostics['records'] = len(records)
    diagnostics['remaining_records'] = len(records)
    _log_source_hub_diagnostics(diagnostics)
    if not records:
        return {
            'assignments': [],
            'clusters': 0,
            'diagnostics': diagnostics,
            'hubs': [],
            'records': 0,
        }
    if source is not None and any(
        record.get('source') != source for record in records
    ):
        raise ValueError(f'event component is outside source {source!r}')
    missing = [record for record in records if not record.get('embedding')]
    if missing:
        names = [record.get('name') for record in missing[:5]]
        raise RuntimeError(
            f'event hubs: {len(missing)} record(s) lack an embedding — '
            f'run the component enrichment pass first: {names}'
        )
    stage = getattr(config.get_settings().stages, spec.stage_name)
    unassigned = sorted(
        (dict(record) for record in records), key=lambda r: r['uuid']
    )
    clusters: list[list[dict]] = []
    while unassigned:
        pivot = unassigned.pop(0)
        automatic_members: list[dict] = []
        ambiguous: list[tuple[float, dict]] = []
        for candidate in unassigned:
            diagnostics['embedding_comparisons'] += 1
            score = embeddings.cosine_similarity(
                pivot['embedding'], candidate['embedding']
            )
            if score >= stage.merge_above:
                automatic_members.append(candidate)
            elif score > stage.separate_below:
                ambiguous.append((score, candidate))
        ambiguous.sort(key=lambda item: (-item[0], item[1]['uuid']))
        diagnostics['ambiguous_candidates'] += len(ambiguous)
        candidate_limit = getattr(stage, 'rerank_candidate_limit', 32)
        admitted = [candidate for _, candidate in ambiguous[:candidate_limit]]
        reranked_members = await _rerank_ambiguous_candidates(
            pivot,
            admitted,
            comparison_token_budget=stage.comparison_token_budget,
            rerank_top_n=stage.rerank_top_n,
            diagnostics=diagnostics,
        )
        cluster = [pivot, *automatic_members, *reranked_members]
        member_uuids = {member['uuid'] for member in cluster}
        unassigned = [
            candidate
            for candidate in unassigned
            if candidate['uuid'] not in member_uuids
        ]
        clusters.append(cluster)
        diagnostics['locked_groups'] += 1
        diagnostics['remaining_records'] = len(unassigned)
        _log_source_hub_diagnostics(
            diagnostics, group=diagnostics['locked_groups']
        )
    diagnostics['synthesizer_calls'] = len(clusters)
    _log_source_hub_diagnostics(diagnostics)
    definitions = await _synthesize_definitions(
        clusters, synthesizer, llm.gate(max_concurrency), spec.synthesis_context
    )
    hub_vectors = await embeddings.embedder().embed(
        [
            f'{definition["canonical_name"]}: {definition["description"]}'
            for definition in definitions
        ]
    )
    hub_records: list[dict] = []
    assignments: list[dict] = []
    for cluster, definition, embedding in zip(
        clusters, definitions, hub_vectors, strict=True
    ):
        hub_uuid = spec.hub_id_factory(cluster, definition)
        members = sorted(member['uuid'] for member in cluster)
        hub_records.append(
            {
                'uuid': hub_uuid,
                'source': spec.source_resolver(cluster),
                'canonical_name': definition['canonical_name'],
                'aliases': sorted(
                    {
                        alias
                        for member in cluster
                        for alias in [
                            member['name'],
                            *member.get('aliases', []),
                        ]
                        if alias
                    }
                ),
                'description': definition['description'],
                'embedding': embedding,
                'members': members,
            }
        )
        assignments.extend(
            {'component': member_uuid, 'hub': hub_uuid}
            for member_uuid in members
        )
    diagnostics['final_hubs'] = len(hub_records)
    _log_source_hub_diagnostics(diagnostics)
    return {
        'assignments': sorted(assignments, key=lambda item: item['component']),
        'clusters': len(clusters),
        'diagnostics': diagnostics,
        'hubs': hub_records,
        'records': len(records),
    }


async def rebuild(
    source: str,
    *,
    session_factory,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild source-local event hubs from persisted components."""
    spec = EventHubSpec(
        graph=graph_event_hubs,
        stage_name='event_hubs',
        hub_id_factory=source_hub_id_factory(graph_event_hubs),
        source_resolver=lambda records: source or _dominant_source(records),
        synthesis_context=EVENT_SPEC.synthesis_context,
        all_components=queries.all_event_components,
        clear_hubs=writer.clear_event_hubs,
        persist_hubs=writer.persist_event_hubs,
    )
    component_rows = await queries.all_event_components(session_factory, source)
    result = await build_event_hubs(
        component_rows,
        source=source,
        spec=spec,
        max_concurrency=max_concurrency,
        synthesizer=synthesizer,
    )
    await writer.clear_event_hubs(source, session_factory=session_factory)
    await writer.persist_event_hubs(
        result['hubs'], session_factory=session_factory, tier='source'
    )
    return {'clusters': result['clusters'], 'records': result['records']}


class EventHubNode:
    """Builds source-local EventHubs from enriched event components."""

    def __init__(self, synthesizer) -> None:
        self._synthesizer = synthesizer

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        construction_bundle = state.to_construction_bundle(current_state)
        source = construction_bundle.source.key or ''
        records = [
            {
                'uuid': component.uuid,
                'name': component.name,
                'description': component.description,
                'embedding': component.embedding,
                'source': component.source,
                'aliases': [],
            }
            for component in construction_bundle.event_hub_components
        ]
        result = await build_event_hubs(
            records,
            source=source,
            spec=_workflow_spec(),
            synthesizer=self._synthesizer,
        )
        construction_bundle.event_hub_assignments = result.get(
            'assignments', []
        )
        construction_bundle.event_hub_records = result.get('hubs', [])
        return {
            'event_hub_assignments': construction_bundle.event_hub_assignments,
            'event_hub_diagnostics': result['diagnostics'],
            'event_hub_records': construction_bundle.event_hub_records,
            'construction_bundle': construction_bundle,
        }
