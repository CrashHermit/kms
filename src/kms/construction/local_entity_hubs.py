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
from kms.construction import name_hubs, triplet_hubs
from kms.core import (
    batching,
    clustering,
    context_window,
    embeddings,
    llm,
    models,
    module,
    vector_index,
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


async def _rebuild_triplets_callback(
    *, language_model, source=None, session_factory=None, max_concurrency=None
):
    if source is None:
        raise ValueError('source is required for local triplet rebuilding')
    return await triplet_hubs.rebuild(
        language_model=language_model,
        source=source,
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
    adjudication_context: str
    synthesis_context: str
    all_components: callable
    clear_hubs: callable
    persist_hubs: callable
    rebuild_names: callable
    rebuild_triplets: callable


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
    adjudication_context='Compare durable component mentions within one source.',
    synthesis_context='Synthesize a source-local semantic hub.',
    all_components=queries.all_entity_components,
    clear_hubs=writer.clear_entity_hubs,
    persist_hubs=writer.persist_entity_hubs,
    rebuild_names=_rebuild_names_callback,
    rebuild_triplets=_rebuild_triplets_callback,
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


class EntityHubAdjudicationSignature(dspy.Signature):
    r"""
    Compare two entity mentions. Return TRUE only when they are the same
    canonical concept. Return FALSE for broader, narrower, related, composed,
    or part-whole concepts.
    """

    comparison: models.HubMentionComparisonInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


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


class EntityHubAdjudicator(module.Module):
    signature = EntityHubAdjudicationSignature
    record_name = 'entity_hub_adjudicator'

    def encode(
        self,
        left: models.HubMentionInput,
        right: models.HubMentionInput,
        scope: str,
    ) -> dict:
        return {
            'comparison': models.HubMentionComparisonInput(
                left=left, right=right, scope=scope
            )
        }

    def decode(self, prediction, **inputs) -> bool:
        result = models.MergeDecision.model_validate(prediction.result)
        return module.require_bool(result.should_merge, 'should_merge')


def _coarse_clusters(
    mentions: list[dict], recall_threshold: float
) -> list[list[dict]]:
    """Groups mentions into connected components by cosine similarity."""
    count = len(mentions)
    adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
    for i in range(count):
        for j in range(i + 1, count):
            if (
                embeddings.cosine_similarity(
                    mentions[i]['embedding'], mentions[j]['embedding']
                )
                >= recall_threshold
            ):
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited: set[int] = set()
    clusters: list[list[dict]] = []
    for i in range(count):
        if i not in visited:
            component: list[dict] = []
            stack = [i]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    component.append(mentions[vertex])
                    stack.extend(adjacency[vertex])
            clusters.append(component)
    return clusters


def _central_mention(component: list[dict]) -> dict:
    """Returns the mention most similar to the others in a component."""
    if len(component) == 1:
        return component[0]
    best = component[0]
    best_score = -1.0
    for candidate in component:
        others = [member for member in component if member is not candidate]
        score = sum(
            embeddings.cosine_similarity(
                candidate['embedding'], other['embedding']
            )
            for other in others
        ) / len(others)
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _mention_text(record: dict) -> str:
    """Renders a hub mention record as compact text for the reranker."""
    mention = models.HubMentionInput.from_record(record)
    parts = [mention.name]
    parts.extend(mention.aliases)
    if mention.description:
        parts.append(mention.description)
    return ' '.join(parts)


async def _adjudicate_component(
    component: list[dict],
    adjudicator: Any,
    merge_above: float,
    separate_below: float,
    gate: asyncio.Semaphore,
    scope: str,
    comparison_token_budget: int = 4096,
    rerank_top_n: int | None = None,
) -> list[list[dict]]:
    """Splits one coarse component into final clusters."""
    unassigned = list(component)
    clusters: list[list[dict]] = []
    boundary_pairs = 0
    wave_count = 0
    largest_wave_tokens = 0

    while unassigned:
        pivot = _central_mention(unassigned)
        unassigned = [member for member in unassigned if member is not pivot]
        if rerank_top_n:
            selected = await clustering.select_reranked_candidates(
                _mention_text(pivot),
                unassigned,
                _mention_text,
                top_n=rerank_top_n,
            )
            selected_ids = {id(member) for member in selected}
            omitted = [
                member
                for member in unassigned
                if id(member) not in selected_ids
            ]
            unassigned = selected
        else:
            omitted = []
        pivot_cluster = [pivot]
        clusters.append(pivot_cluster)
        remaining = list(omitted)
        boundary_candidates: list[dict] = []
        for member in unassigned:
            score = embeddings.cosine_similarity(
                pivot['embedding'], member['embedding']
            )
            if score >= merge_above:
                pivot_cluster.append(member)
            elif score > separate_below:
                boundary_candidates.append(member)
            else:
                remaining.append(member)

        def token_cost(member: dict, pivot_record: dict = pivot) -> int:
            left = models.HubMentionInput.from_record(pivot_record)
            right = models.HubMentionInput.from_record(member)
            return (
                context_window.estimate_text_tokens(left.name)
                + sum(
                    context_window.estimate_text_tokens(alias)
                    for alias in left.aliases
                )
                + context_window.estimate_text_tokens(left.description)
                + context_window.estimate_text_tokens(right.name)
                + sum(
                    context_window.estimate_text_tokens(alias)
                    for alias in right.aliases
                )
                + context_window.estimate_text_tokens(right.description)
                + 32
            )

        boundary_pairs += len(boundary_candidates)
        for wave in batching.token_batches(
            boundary_candidates,
            token_cost=token_cost,
            token_budget=comparison_token_budget,
        ):
            wave_count += 1
            wave_tokens = sum(token_cost(member) for member in wave)
            largest_wave_tokens = max(largest_wave_tokens, wave_tokens)

            async def decide(member: dict, pivot_record: dict = pivot) -> Any:
                async with gate:
                    return await adjudicator.aforward(
                        left=models.HubMentionInput.from_record(pivot_record),
                        right=models.HubMentionInput.from_record(member),
                        scope=scope,
                    )

            decisions = await asyncio.gather(
                *(decide(member) for member in wave)
            )
            for member, decision in zip(wave, decisions, strict=True):
                if decision:
                    pivot_cluster.append(member)
                else:
                    remaining.append(member)
        unassigned = remaining

    logger.info(
        'entity hub comparisons: %d boundary pairs, %d waves, '
        'largest wave %d estimated tokens',
        boundary_pairs,
        wave_count,
        largest_wave_tokens,
    )
    return clusters


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


async def build_hubs(
    records: list[dict],
    *,
    spec: EntityHubSpec,
    recall_threshold: float | None = None,
    merge_above: float | None = None,
    separate_below: float | None = None,
    max_concurrency: int | None = None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Builds reusable entity hubs from already-loaded source records."""
    stage = getattr(config.get_settings().stages, spec.stage_name)
    if recall_threshold is None:
        recall_threshold = stage.recall_threshold
    if merge_above is None:
        merge_above = stage.merge_above
    if separate_below is None:
        separate_below = stage.separate_below

    if not records:
        return {
            'clusters': 0,
            'records': 0,
            'hubs': [],
        }

    missing = [record for record in records if not record.get('embedding')]
    if missing:
        names = [record.get('name') for record in missing[:5]]
        raise RuntimeError(
            f'entity hubs: {len(missing)} record(s) lack an '
            f'embedding — run the component enrichment pass first: {names}'
        )

    components = _coarse_clusters(records, recall_threshold)
    logger.info(
        'entity hubs: %d coarse component(s) at recall %.2f',
        len(components),
        recall_threshold,
    )

    gate = llm.gate(max_concurrency)
    clusters: list[list[dict]] = []
    for component in components:
        local_clusters = await _adjudicate_component(
            component,
            adjudicator,
            merge_above,
            separate_below,
            gate,
            spec.adjudication_context,
            stage.comparison_token_budget,
            stage.rerank_top_n,
        )
        clusters.extend(local_clusters)
    definitions = await _synthesize_definitions(
        clusters, synthesizer, gate, spec.synthesis_context
    )
    hub_texts = [
        f'{definition["canonical_name"]}: {definition["description"]}'
        for definition in definitions
    ]
    hub_vectors = await embeddings.embedder().embed(
        [text for text in hub_texts]
    )

    make_hub_id = spec.hub_id_factory
    cluster_uuids = [
        make_hub_id(cluster, definition)
        for cluster, definition in zip(clusters, definitions, strict=True)
    ]
    by_uuid: dict[str, list[int]] = {}
    for index, cluster_uuid in enumerate(cluster_uuids):
        by_uuid.setdefault(cluster_uuid, []).append(index)
    coalesced: list[list[int]] = list(by_uuid.values())

    hub_records: list[dict] = []
    for indices in coalesced:
        members = [member for index in indices for member in clusters[index]]
        primary = indices[0]
        definition = definitions[primary]
        hub_records.append(
            {
                'uuid': cluster_uuids[primary],
                'source': spec.source_resolver(members),
                'canonical_name': definition['canonical_name'],
                'aliases': sorted(
                    {
                        alias
                        for member in members
                        for alias in [
                            member['name'],
                            *member.get('aliases', []),
                        ]
                        if alias
                    }
                ),
                'description': definition['description'],
                'embedding': hub_vectors[primary],
                'members': [member['uuid'] for member in members],
            }
        )

    return {
        'clusters': len(clusters),
        'records': len(records),
        'hubs': hub_records,
    }


async def _choose_hub(
    spec: EntityHubSpec,
    record: dict,
    candidates: list[dict],
    adjudicator: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> tuple[str | None, list[tuple[str, str]], float, str]:
    stage = getattr(config.get_settings().stages, spec.stage_name)
    for candidate in candidates:
        score = candidate['score']
        if score >= stage.merge_above:
            return candidate['uuid'], [], score, 'Merge'
        if score <= stage.separate_below:
            continue
        async with gate:
            decision = await adjudicator.aforward(
                left=models.HubMentionInput.from_record(record),
                right=models.HubMentionInput.from_record(candidate),
                scope=scope,
            )
        if decision:
            return candidate['uuid'], [], score, 'Merge'
    return None, [], 0.0, 'Separate'


async def _new_hub(
    record: dict,
    synthesizer: Any,
    gate: asyncio.Semaphore,
    spec: EntityHubSpec,
) -> dict:
    definition = (
        await _synthesize_definitions(
            [[record]], synthesizer, gate, spec.synthesis_context
        )
    )[0]
    vector = await embeddings.embedder().embed(
        [f'{definition["canonical_name"]}: {definition["description"]}']
    )
    return {
        'uuid': spec.hub_id_factory([record], definition),
        'source': spec.source_resolver([record]),
        'canonical_name': definition['canonical_name'],
        'aliases': sorted(
            {
                value
                for value in [record.get('name'), *record.get('aliases', [])]
                if value
            }
        ),
        'description': definition['description'],
        'embedding': vector[0],
        'members': [record['uuid']],
    }


async def _refresh_local_hubs(
    spec: EntityHubSpec,
    records_by_hub: dict[str, list[dict]],
    new_hub_ids: set[str],
    synthesizer: Any,
    gate: asyncio.Semaphore,
) -> list[dict]:
    refreshed: list[dict] = []
    for hub_uuid, records in records_by_hub.items():
        definition = (
            await _synthesize_definitions(
                [records],
                synthesizer,
                gate,
                spec.synthesis_context,
            )
        )[0]
        vector = await embeddings.embedder().embed(
            [f'{definition["canonical_name"]}: {definition["description"]}']
        )
        hub = {
            'uuid': hub_uuid,
            'source': spec.source_resolver(records),
            'canonical_name': definition['canonical_name'],
            'aliases': sorted(
                {alias for record in records for alias in _aliases(record)}
            ),
            'description': definition['description'],
            'embedding': vector[0],
        }
        if hub_uuid in new_hub_ids:
            hub['members'] = [record['uuid'] for record in records]
        refreshed.append(hub)
    return refreshed


def _aliases(record: dict) -> set[str]:
    name = record.get('name') or record.get('canonical_name')
    return {value for value in [name, *record.get('aliases', [])] if value}


async def assign_local_hubs(
    bundle,
    *,
    max_concurrency: int | None = None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Assign entity components to source-local EntityHubs."""
    spec = EntityHubSpec(
        graph=hubs,
        stage_name=ENTITY_SPEC.stage_name,
        hub_id_factory=lambda records, definition: hubs.hub_uuid(
            hubs.LOCAL_ENTITY_HUB_LABEL,
            max(
                (r.get('source') for r in records),
                key=lambda source: sum(
                    1 for record in records if record.get('source') == source
                ),
            ),
            '|'.join(sorted(record['uuid'] for record in records)),
        ),
        source_resolver=ENTITY_SPEC.source_resolver,
        adjudication_context=ENTITY_SPEC.adjudication_context,
        synthesis_context=ENTITY_SPEC.synthesis_context,
        all_components=ENTITY_SPEC.all_components,
        clear_hubs=ENTITY_SPEC.clear_hubs,
        persist_hubs=ENTITY_SPEC.persist_hubs,
        rebuild_names=ENTITY_SPEC.rebuild_names,
        rebuild_triplets=ENTITY_SPEC.rebuild_triplets,
    )

    source = bundle.source
    if any(record.source != source for record in bundle.components):
        raise ValueError(f'entity component is outside source {source!r}')
    if any(record.source != source for record in bundle.candidate_hubs):
        raise ValueError(f'entity candidate hub is outside source {source!r}')
    records = [
        {
            'uuid': record.uuid,
            'name': record.name,
            'aliases': list(record.aliases),
            'description': record.description,
            'embedding': record.embedding,
            'source': record.source,
        }
        for record in bundle.components
    ]
    candidate_by_uuid = {
        record.uuid: {
            'uuid': record.uuid,
            'name': record.name,
            'aliases': list(record.aliases),
            'description': record.description,
            'embedding': record.embedding,
            'source': source,
        }
        for record in bundle.candidate_hubs
    }
    dimension = (
        len(bundle.candidate_hubs[0].embedding)
        if bundle.candidate_hubs
        else len(records[0]['embedding'])
        if records
        else 1
    )
    candidate_index = vector_index.ExactCosineIndex(dimension)
    candidate_index.add(
        [(record.uuid, record.embedding) for record in bundle.candidate_hubs]
    )
    if not records:
        return {
            'assigned': 0,
            'new_hubs': 0,
            'changed_hubs': [],
        }
    if any(not record.get('embedding') for record in records):
        names = [
            record['name'] for record in records if not record.get('embedding')
        ][:5]
        raise RuntimeError(
            f'entity hubs: unassigned record(s) lack an '
            f'embedding '
            f'— run the component enrichment pass first: {names}'
        )

    top_k = config.get_settings().stages.search.top_k
    stage = getattr(config.get_settings().stages, spec.stage_name)
    gate = llm.gate(max_concurrency)
    new_hubs: dict[str, dict] = {}
    assignments: list[dict] = []
    aliases: dict[str, set[str]] = {}
    records_by_hub: dict[str, list[dict]] = {}
    for record in records:
        candidates = [
            {
                **candidate_by_uuid[match.key],
                'score': match.score,
            }
            for match in candidate_index.search(
                record['embedding'], top_k=top_k
            )
        ]
        candidates.extend(
            {
                **hub,
                'name': hub['canonical_name'],
                'aliases': hub.get('aliases') or [],
                'source': source,
                'score': embeddings.cosine_similarity(
                    record['embedding'], hub['embedding']
                ),
            }
            for hub in new_hubs.values()
        )
        candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
        candidates = await clustering.select_reranked_candidates(
            _mention_text(record),
            candidates,
            _mention_text,
            top_n=stage.rerank_top_n,
        )
        merged_hub, _, _, _ = await _choose_hub(
            spec,
            record,
            candidates,
            adjudicator,
            gate,
            spec.adjudication_context,
        )

        if merged_hub is None:
            hub = await _new_hub(record, synthesizer, gate, spec)
            merged_hub = hub['uuid']
            existing = new_hubs.setdefault(merged_hub, hub)
            existing['members'] = list(
                dict.fromkeys(existing['members'] + hub['members'])
            )
            existing['aliases'] = sorted(
                set(existing['aliases']) | set(hub['aliases'])
            )
            if merged_hub not in candidate_by_uuid:
                candidate_by_uuid[merged_hub] = {
                    **existing,
                    'name': existing['canonical_name'],
                }
                candidate_index.add([(merged_hub, existing['embedding'])])

        assignments.append({'component': record['uuid'], 'hub': merged_hub})
        aliases.setdefault(merged_hub, set())
        aliases[merged_hub].update(_aliases(record))
        for candidate in candidates:
            if candidate['uuid'] == merged_hub:
                aliases[merged_hub].update(_aliases(candidate))
                if merged_hub not in new_hubs:
                    candidate_record = dict(candidate)
                    candidate_record.pop('score', None)
                    records_by_hub.setdefault(merged_hub, [candidate_record])
                break
        records_by_hub.setdefault(merged_hub, []).append(record)

    refreshed_hubs = await _refresh_local_hubs(
        spec,
        records_by_hub,
        set(new_hubs),
        synthesizer,
        gate,
    )
    result = {
        'assigned': len(assignments),
        'new_hubs': len(new_hubs),
        'changed_hubs': sorted(aliases),
    }
    result.update(
        assignments=assignments,
        hubs=refreshed_hubs,
    )
    return result


async def rebuild(
    source: str,
    *,
    session_factory,
    language_model,
    adjudicator,
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
        adjudication_context='Compare durable component mentions within one source.',
        synthesis_context='Synthesize a source-local semantic hub.',
        all_components=queries.all_entity_components,
        clear_hubs=writer.clear_entity_hubs,
        persist_hubs=writer.persist_entity_hubs,
        rebuild_names=_rebuild_names_callback,
        rebuild_triplets=_rebuild_triplets_callback,
    )
    component_rows = await queries.all_entity_components(
        session_factory, source
    )
    result = await build_hubs(
        component_rows,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
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
    await _rebuild_triplets_callback(
        language_model=language_model,
        source=source,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {'clusters': result['clusters'], 'records': result['records']}


class EntityHubNode:
    def __init__(
        self,
        adjudicator,
        synthesizer,
    ) -> None:
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        construction_bundle = state.to_construction_bundle(current_state)

        # Inline the HubInputNode conversion
        source = construction_bundle.source.key or ''
        entity_bundle = models.HubBuildBundle(
            source=source,
            components=tuple(
                models.HubRecord(
                    uuid=component.uuid,
                    name=component.name,
                    description=component.description,
                    embedding=component.embedding,
                    source=component.source,
                    aliases=(),
                )
                for component in construction_bundle.entity_hub_components
            ),
            candidate_hubs=tuple(
                models.HubRecord(
                    uuid=record['uuid'],
                    name=record.get('canonical_name', record.get('name', '')),
                    description=record.get('description'),
                    embedding=list(record['embedding']),
                    source=record.get('source', ''),
                    aliases=tuple(record.get('aliases') or []),
                )
                for record in construction_bundle.entity_hub_records
            ),
        )

        result = await assign_local_hubs(
            entity_bundle,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        construction_bundle.entity_hub_assignments = result.get(
            'assignments', []
        )
        construction_bundle.entity_hub_records = result.get('hubs', [])
        return {
            'entity_hub_assignments': construction_bundle.entity_hub_assignments,
            'entity_hub_records': construction_bundle.entity_hub_records,
            'construction_bundle': construction_bundle,
        }
