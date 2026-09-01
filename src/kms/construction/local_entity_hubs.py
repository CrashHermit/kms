"""Build reusable EntityHub concepts from source-local components.

Source-level triplet components carry local names and passage-grounded glosses.
This module groups those mentions into reusable EntityHub concepts while
preserving source provenance and supported meaning.
"""

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.construction import name_hubs
from kms.core import (
    batching,
    context_window,
    embeddings,
    llm,
    models,
    module,
    reranker,
)
from kms.graph import (
    hubs,
    queries,
    writer,
)
from kms.graph import (
    local_entity_hubs as graph_entity_hubs,
)

logger = logging.getLogger(__name__)


async def _rebuild_names_callback(
    domain: str,
    source: str,
    *,
    language_model,
    session_factory,
    max_concurrency=None,
):
    return await name_hubs.rebuild(
        domain,
        source,
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )


@dataclass
class EntityHubSpec:
    """Fixed entity-hub metadata consumed by the builder."""

    graph: Any
    stage_name: str
    hub_id_factory: callable
    source_resolver: callable
    synthesis_context: str
    all_components: callable
    clear_hubs: callable
    persist_hubs: callable
    rebuild_names: callable


ENTITY_SPEC = EntityHubSpec(
    graph=None,  # filled at runtime
    stage_name='entity_hubs',
    hub_id_factory=lambda records, definition, graph: graph.hub_uuid(
        max(
            (r.get('source') for r in records),
            key=lambda s: sum(1 for r in records if r.get('source') == s),
        ),
        '|'.join(sorted(r['uuid'] for r in records)),
    ),
    source_resolver=lambda records: max(
        (r.get('source') for r in records),
        key=lambda s: sum(1 for r in records if r.get('source') == s),
    ),
    synthesis_context='Synthesize a source-local semantic hub.',
    all_components=queries.all_entity_components,
    clear_hubs=writer.clear_entity_hubs,
    persist_hubs=writer.persist_entity_hubs,
    rebuild_names=_rebuild_names_callback,
)


class EntityHubDefinition(BaseModel):
    canonical_name: str = Field(
        description='The canonical name for the entity concept.'
    )
    description: str = Field(
        description='A standalone canonical entity description supported by the supplied evidence.'
    )


class EntityHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one source-local canonical entity concept from supplied
    surface forms and passage-grounded descriptions. Generalize only what
    evidence supports. Do not invent facts.
    """

    request: models.HubSynthesisInput = dspy.InputField()
    result: EntityHubDefinition = dspy.OutputField()


class EntityHubSynthesizer(module.Module):
    signature = EntityHubSynthesisSignature
    record_name = 'entity_hub_synthesizer'

    def encode(
        self,
        surface_forms: list[str],
        descriptions: list[str],
        scope: str,
    ) -> dict:
        return {
            'request': models.HubSynthesisInput(
                surface_forms=surface_forms,
                descriptions=descriptions,
                scope=scope,
            )
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        if not isinstance(prediction.result, EntityHubDefinition):
            raise TypeError('result must be an EntityHubDefinition value')
        result = EntityHubDefinition.model_validate(prediction.result)
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
    diagnostics: dict[str, int],
    *,
    group: int | None = None,
) -> None:
    """Logs cumulative source-local hub build diagnostics."""
    details: dict[str, int] = dict(diagnostics)
    if group is not None:
        details['group'] = group
    logger.info('entity hub diagnostics: %s', details)


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
            f'entity hubs: pivot {pivot["uuid"]!r} requires '
            f'{pivot_tokens} estimated tokens; reranker budget is '
            f'{comparison_token_budget}'
        )
    candidate_texts = [
        (candidate, _mention_text(candidate)) for candidate in candidates
    ]
    for candidate, text in candidate_texts:
        token_cost = context_window.estimate_text_tokens(text)
        if pivot_tokens + token_cost > comparison_token_budget:
            raise ValueError(
                f'entity hubs: candidate {candidate["uuid"]!r} with pivot '
                f'{pivot["uuid"]!r} requires '
                f'{pivot_tokens + token_cost} estimated tokens; reranker '
                f'budget is {comparison_token_budget}'
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
        results = await client.rerank(
            pivot_text,
            documents,
            top_n=rerank_top_n,
        )
        for result in results:
            index = result['index']
            if 0 <= index < len(batch):
                diagnostics['reranker_selected'] += 1
                candidate = batch[index][0]
                if candidate['uuid'] not in selected_uuids:
                    selected.append(candidate)
                    selected_uuids.add(candidate['uuid'])
    return selected


def _mention_text(record: dict) -> str:
    """Renders a hub mention record as compact text for the reranker."""
    mention = models.HubMentionInput.from_record(record)
    parts = [mention.name]
    parts.extend(mention.aliases)
    if mention.description:
        parts.append(mention.description)
    return ' '.join(parts)


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

    results = await asyncio.gather(*(_one(cluster) for cluster in clusters))
    return list(results)


def _dominant_source(records: list[dict]) -> str:
    """Returns the most common source represented by a group of records."""
    counts = Counter(
        record['source'] for record in records if record.get('source')
    )
    return counts.most_common(1)[0][0] if counts else 'unknown'


def source_hub_id_factory(graph: Any) -> callable:
    """Creates a deterministic source-hub id function for entity hubs."""

    def make_id(records: list[dict], definition: dict) -> str:
        identity = '|'.join(sorted(record['uuid'] for record in records))
        return graph.hub_uuid(_dominant_source(records), identity)

    return make_id


def _workflow_spec() -> EntityHubSpec:
    """Returns the graph-specific metadata for workflow entity hubs."""
    return EntityHubSpec(
        graph=hubs,
        stage_name=ENTITY_SPEC.stage_name,
        hub_id_factory=lambda records, definition: hubs.hub_uuid(
            hubs.LOCAL_ENTITY_HUB_LABEL,
            _dominant_source(records),
            '|'.join(sorted(record['uuid'] for record in records)),
        ),
        source_resolver=ENTITY_SPEC.source_resolver,
        synthesis_context=ENTITY_SPEC.synthesis_context,
        all_components=ENTITY_SPEC.all_components,
        clear_hubs=ENTITY_SPEC.clear_hubs,
        persist_hubs=ENTITY_SPEC.persist_hubs,
        rebuild_names=ENTITY_SPEC.rebuild_names,
    )


async def build_entity_hubs(
    records: list[dict],
    *,
    source: str | None,
    spec: EntityHubSpec,
    max_concurrency: int | None = None,
    synthesizer: Any,
) -> dict:
    """Clusters all source-local entity components into locked hub groups."""
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
        raise ValueError(f'entity component is outside source {source!r}')
    missing = [record for record in records if not record.get('embedding')]
    if missing:
        names = [record.get('name') for record in missing[:5]]
        raise RuntimeError(
            f'entity hubs: {len(missing)} record(s) lack an '
            f'embedding — run the component enrichment pass first: {names}'
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
                pivot['embedding'],
                candidate['embedding'],
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
            diagnostics,
            group=diagnostics['locked_groups'],
        )

    diagnostics['synthesizer_calls'] = len(clusters)
    _log_source_hub_diagnostics(diagnostics)
    definitions = await _synthesize_definitions(
        clusters,
        synthesizer,
        llm.gate(max_concurrency),
        spec.synthesis_context,
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
        clusters,
        definitions,
        hub_vectors,
        strict=True,
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
    language_model,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild source-local entity hubs from persisted components."""
    spec = EntityHubSpec(
        graph=graph_entity_hubs,
        stage_name='entity_hubs',
        hub_id_factory=source_hub_id_factory(graph_entity_hubs),
        source_resolver=lambda records: max(
            (r.get('source') for r in records),
            key=lambda s: sum(1 for r in records if r.get('source') == s),
        ),
        synthesis_context='Synthesize a source-local semantic hub.',
        all_components=queries.all_entity_components,
        clear_hubs=writer.clear_entity_hubs,
        persist_hubs=writer.persist_entity_hubs,
        rebuild_names=_rebuild_names_callback,
    )
    component_rows = await queries.all_entity_components(
        session_factory, source
    )
    result = await build_entity_hubs(
        component_rows,
        source=source,
        spec=spec,
        max_concurrency=max_concurrency,
        synthesizer=synthesizer,
    )
    await writer.clear_entity_hubs(source, session_factory=session_factory)
    await writer.persist_entity_hubs(
        result['hubs'],
        session_factory=session_factory,
        tier='source',
    )
    await _rebuild_names_callback(
        'entity',
        source,
        language_model=language_model,
        session_factory=session_factory,
    )
    return result


class EntityHubNode:
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
            for component in construction_bundle.entity_hub_components
        ]
        result = await build_entity_hubs(
            records,
            source=source,
            spec=_workflow_spec(),
            synthesizer=self._synthesizer,
        )
        construction_bundle.entity_hub_assignments = result.get(
            'assignments', []
        )
        construction_bundle.entity_hub_records = result.get('hubs', [])
        return {
            'entity_hub_assignments': construction_bundle.entity_hub_assignments,
            'entity_hub_diagnostics': result['diagnostics'],
            'entity_hub_records': construction_bundle.entity_hub_records,
            'construction_bundle': construction_bundle,
        }
