"""Build reusable PredicateHub concepts from source-local components.

Source-level triplet components carry local names and passage-grounded glosses.
This module builds those mentions into reusable PredicateHub concepts, preserving
source provenance while synthesizing standalone learner-facing descriptions.
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
from kms.core import content, embeddings, llm, models, module, vector_index
from kms.graph import hubs, queries, writer

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


async def _rebuild_meta_names_callback(
    domain: str, *, language_model, session_factory, max_concurrency=None
):
    return await name_hubs.rebuild_meta(
        domain,
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )


async def _rebuild_triplets_callback(
    *, language_model, source=None, session_factory=None, max_concurrency=None
):
    if source is not None:
        return await triplet_hubs.rebuild(
            language_model=language_model,
            source=source,
            session_factory=session_factory,
            max_concurrency=max_concurrency,
        )
    return await triplet_hubs.rebuild_meta(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )


@dataclass(frozen=True)
class PredicateHubSpec:
    """Fixed predicate-hub metadata consumed by the builder."""

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


PREDICATE_SPEC = PredicateHubSpec(
    graph=None,  # filled at runtime
    stage_name='predicate_hubs',
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
    all_components=queries.all_predicate_components,
    clear_hubs=writer.clear_predicate_hubs,
    persist_hubs=writer.persist_predicate_hubs,
    rebuild_names=_rebuild_names_callback,
    rebuild_triplets=_rebuild_triplets_callback,
)


class PredicateHubDefinition(BaseModel):
    canonical_name: str = Field(
        description='The canonical name for the predicate relation.'
    )
    description: str = Field(
        description='A standalone learner-facing predicate description.'
    )


class PredicateHubAdjudication(BaseModel):
    decision: str = Field(
        description='Merge, Hierarchy, or Separate for predicate mentions.'
    )
    more_general: str = Field(
        default='none',
        description='left or right when the decision is Hierarchy.',
    )


class PredicateHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one source-local learner-facing predicate concept from supplied
    relation phrases and their passage-grounded descriptions. Generalize only
    what the evidence supports. Do not mention the source or invent facts.
    """

    surface_forms: list[str] = dspy.InputField()
    descriptions: list[str] = dspy.InputField()
    scope: str = dspy.InputField()
    result: PredicateHubDefinition = dspy.OutputField()


class PredicateHubAdjudicationSignature(dspy.Signature):
    r"""
    Compare two predicate mentions. Return Merge for the same relation,
    Hierarchy only for a strict kind-of relation, and Separate otherwise.
    """

    left: content.ContentParts = dspy.InputField()
    right: content.ContentParts = dspy.InputField()
    scope: str = dspy.InputField()
    result: PredicateHubAdjudication = dspy.OutputField()


class PredicateHubSynthesizer(module.Module):
    signature = PredicateHubSynthesisSignature
    record_name = 'predicate_hub_synthesizer'

    def encode(
        self,
        surface_forms: list[str],
        descriptions: list[str],
        scope: str,
    ) -> dict:
        return {
            'surface_forms': surface_forms,
            'descriptions': descriptions,
            'scope': scope,
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


class PredicateHubAdjudicator(module.Module):
    signature = PredicateHubAdjudicationSignature
    record_name = 'predicate_hub_adjudicator'

    def encode(
        self,
        left: content.Content,
        right: content.Content,
        scope: str,
    ) -> dict:
        return {
            'left': content.ContentParts(content=left),
            'right': content.ContentParts(content=right),
            'scope': scope,
        }

    def decode(self, prediction, **inputs) -> PredicateHubAdjudication:
        result = prediction.result
        if not isinstance(result, PredicateHubAdjudication):
            raise TypeError(
                'result must be a PredicateHubAdjudication value'
            )
        return result


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


def _mention_content(mention: dict) -> content.Content:
    """Builds the adjudicator's content view of one hub-building record."""
    parts: list[content.TextPart] = [
        content.TextPart(text=f'Name: {mention["name"]}'),
    ]
    aliases = [
        alias
        for alias in mention.get('aliases', [])
        if alias != mention['name']
    ]
    if aliases:
        parts.append(content.TextPart(text=f'Aliases: {", ".join(aliases)}'))
    if mention.get('description'):
        parts.append(
            content.TextPart(text=f'Description: {mention["description"]}')
        )
    return content.Content(parts=parts)


async def _adjudicate_component(
    component: list[dict],
    adjudicator: Any,
    merge_above: float,
    separate_below: float,
    gate: asyncio.Semaphore,
    scope: str,
) -> tuple[list[list[dict]], list[tuple[int, int]]]:
    """Splits one coarse component into final clusters."""
    unassigned = list(component)
    clusters: list[list[dict]] = []
    subsumption: list[tuple[int, int]] = []

    while unassigned:
        pivot = _central_mention(unassigned)
        unassigned = [member for member in unassigned if member is not pivot]
        pivot_cluster = [pivot]
        pivot_index = len(clusters)
        clusters.append(pivot_cluster)

        remaining: list[dict] = []
        for member in unassigned:
            score = embeddings.cosine_similarity(
                pivot['embedding'], member['embedding']
            )
            if score >= merge_above:
                pivot_cluster.append(member)
            elif score <= separate_below:
                remaining.append(member)
            else:
                async with gate:
                    decision = await adjudicator.aforward(
                        left=_mention_content(pivot),
                        right=_mention_content(member),
                        scope=scope,
                    )
                if decision.decision == 'Merge':
                    pivot_cluster.append(member)
                elif decision.decision == 'Hierarchy':
                    member_index = len(clusters)
                    clusters.append([member])
                    if decision.more_general == 'right':
                        subsumption.append((member_index, pivot_index))
                    else:
                        subsumption.append((pivot_index, member_index))
                else:
                    remaining.append(member)
        unassigned = remaining

    return clusters, subsumption


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
    """Creates a deterministic source-hub id function for predicate hubs."""

    def make_id(records: list[dict], definition: dict) -> str:
        identity = '|'.join(sorted(record['uuid'] for record in records))
        return graph.hub_uuid(_dominant_source(records), identity)

    return make_id


def meta_hub_id_factory(graph: Any) -> callable:
    """Creates a deterministic meta-hub id function for predicate hubs."""

    def make_id(records: list[dict], definition: dict) -> str:
        member_ids = sorted(record['uuid'] for record in records)
        if not member_ids:
            raise ValueError('meta hub clusters must contain source hubs')
        return graph.meta_hub_uuid('|'.join(member_ids))

    return make_id


def component_records(rows: list[dict]) -> list[dict]:
    return [
        {
            'uuid': row['uuid'],
            'name': row['name'],
            'aliases': row.get('aliases') or [],
            'description': row.get('description'),
            'embedding': row.get('embedding'),
            'source': row['source'],
        }
        for row in rows
    ]


def source_hub_records(rows: list[dict]) -> list[dict]:
    return [
        {
            'uuid': row['uuid'],
            'name': row['canonical_name'],
            'aliases': row.get('aliases') or [],
            'description': row.get('description'),
            'embedding': row.get('embedding'),
            'source': row.get('source'),
        }
        for row in rows
    ]


def require_meta_sources(
    spec: PredicateHubSpec, records: list[dict]
) -> set[str]:
    sources = {record['source'] for record in records if record.get('source')}
    if len(sources) < 2:
        raise RuntimeError(
            f'predicate hubs (meta): requires at least two distinct '
            f'sources, found {len(sources)}'
        )
    return sources


def _qualify_meta_result(result: dict, records: list[dict]) -> dict:
    source_by_member = {
        record['uuid']: record.get('source') for record in records
    }
    qualifying_hubs = []
    qualifying_ids = set()
    for hub in result['hubs']:
        sources = {
            source_by_member[member]
            for member in hub.get('members', [])
            if source_by_member.get(member)
        }
        if len(sources) >= 2:
            qualifying_hubs.append(hub)
            qualifying_ids.add(hub['uuid'])
    return {
        **result,
        'hubs': qualifying_hubs,
        'subsumption_edges': [
            edge
            for edge in result['subsumption_edges']
            if edge['general'] in qualifying_ids
            and edge['specific'] in qualifying_ids
        ],
    }


async def build_hubs(
    records: list[dict],
    *,
    spec: PredicateHubSpec,
    recall_threshold: float | None = None,
    merge_above: float | None = None,
    separate_below: float | None = None,
    max_concurrency: int | None = None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Builds reusable predicate hubs from already-loaded source records."""
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
            'subsumption_edges': [],
        }

    missing = [record for record in records if not record.get('embedding')]
    if missing:
        names = [record.get('name') for record in missing[:5]]
        raise RuntimeError(
            f'predicate hubs: {len(missing)} record(s) lack an '
            f'embedding — run the component enrichment pass first: {names}'
        )

    components = _coarse_clusters(records, recall_threshold)
    logger.info(
        'predicate hubs: %d coarse component(s) at recall %.2f',
        len(components),
        recall_threshold,
    )

    gate = llm.gate(max_concurrency)
    clusters: list[list[dict]] = []
    subsumption: list[tuple[int, int]] = []
    for component in components:
        local_clusters, local_subsumption = await _adjudicate_component(
            component,
            adjudicator,
            merge_above,
            separate_below,
            gate,
            spec.adjudication_context,
        )
        offset = len(clusters)
        clusters.extend(local_clusters)
        subsumption.extend(
            (general + offset, specific + offset)
            for general, specific in local_subsumption
        )

    definitions = await _synthesize_definitions(
        clusters, synthesizer, gate, spec.synthesis_context
    )
    hub_texts = [
        f'{definition["canonical_name"]}: {definition["description"]}'
        for definition in definitions
    ]
    hub_vectors = await embeddings.embedder().embed(
        [content.Content.from_text(text) for text in hub_texts]
    )

    make_hub_id = spec.hub_id_factory
    cluster_uuids = [
        make_hub_id(cluster, definition, spec.graph)
        for cluster, definition in zip(clusters, definitions, strict=True)
    ]
    by_uuid: dict[str, list[int]] = {}
    for index, cluster_uuid in enumerate(cluster_uuids):
        by_uuid.setdefault(cluster_uuid, []).append(index)
    coalesced: list[list[int]] = list(by_uuid.values())
    cluster_to_hub: dict[int, int] = {
        index: hub_index
        for hub_index, indices in enumerate(coalesced)
        for index in indices
    }

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

    seen_edges: set[tuple[str, str]] = set()
    subsumption_edges: list[dict] = []
    for general, specific in subsumption:
        general_hub = cluster_to_hub[general]
        specific_hub = cluster_to_hub[specific]
        if general_hub == specific_hub:
            continue
        edge = (
            cluster_uuids[coalesced[general_hub][0]],
            cluster_uuids[coalesced[specific_hub][0]],
        )
        if edge not in seen_edges:
            seen_edges.add(edge)
            subsumption_edges.append({'general': edge[0], 'specific': edge[1]})

    return {
        'clusters': len(clusters),
        'records': len(records),
        'hubs': hub_records,
        'subsumption_edges': subsumption_edges,
    }


async def _choose_hub(
    spec: PredicateHubSpec,
    record: dict,
    candidates: list[dict],
    adjudicator: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> tuple[str | None, list[tuple[str, str]], float, str]:
    stage = getattr(config.get_settings().stages, spec.stage_name)
    hierarchy: list[tuple[str, str]] = []
    for candidate in candidates:
        score = candidate['score']
        if score >= stage.merge_above:
            return candidate['uuid'], hierarchy, score, 'Merge'
        if score <= stage.separate_below:
            continue
        async with gate:
            decision = await adjudicator.aforward(
                left=_mention_content(record),
                right=_mention_content(candidate),
                scope=scope,
            )
        if decision.decision == 'Merge':
            return candidate['uuid'], hierarchy, score, 'Merge'
        if decision.decision == 'Hierarchy':
            hierarchy.append((candidate['uuid'], decision.more_general))
    return None, hierarchy, 0.0, 'Separate'


async def _new_hub(
    record: dict,
    synthesizer: Any,
    gate: asyncio.Semaphore,
    spec: PredicateHubSpec,
) -> dict:
    definition = (
        await _synthesize_definitions(
            [[record]], synthesizer, gate, spec.synthesis_context
        )
    )[0]
    vector = await embeddings.embedder().embed(
        [
            content.Content.from_text(
                f'{definition["canonical_name"]}: {definition["description"]}'
            )
        ]
    )
    return {
        'uuid': spec.hub_id_factory([record], definition, spec.graph),
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


async def _refresh_source_hubs(
    spec: PredicateHubSpec,
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
            [
                content.Content.from_text(
                    f'{definition["canonical_name"]}: {definition["description"]}'
                )
            ]
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


async def assign_source_hubs(
    bundle,
    *,
    max_concurrency: int | None = None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Assign predicate components to source-local PredicateHubs."""
    spec = PredicateHubSpec(
        graph=hubs,
        stage_name=PREDICATE_SPEC.stage_name,
        hub_id_factory=lambda records, definition, graph: graph.hub_uuid(
            hubs.PREDICATE_HUB_LABEL,
            max(
                (r.get('source') for r in records),
                key=lambda source: sum(
                    1 for record in records if record.get('source') == source
                ),
            ),
            '|'.join(sorted(record['uuid'] for record in records)),
        ),
        source_resolver=PREDICATE_SPEC.source_resolver,
        adjudication_context=PREDICATE_SPEC.adjudication_context,
        synthesis_context=PREDICATE_SPEC.synthesis_context,
        all_components=PREDICATE_SPEC.all_components,
        clear_hubs=PREDICATE_SPEC.clear_hubs,
        persist_hubs=PREDICATE_SPEC.persist_hubs,
        rebuild_names=PREDICATE_SPEC.rebuild_names,
        rebuild_triplets=PREDICATE_SPEC.rebuild_triplets,
    )

    source = bundle.source
    if any(record.source != source for record in bundle.components):
        raise ValueError(f'predicate component is outside source {source!r}')
    if any(record.source != source for record in bundle.candidate_hubs):
        raise ValueError(
            f'predicate candidate hub is outside source {source!r}'
        )
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
            'hierarchies': 0,
            'changed_hubs': [],
        }
    if any(not record.get('embedding') for record in records):
        names = [
            record['name'] for record in records if not record.get('embedding')
        ][:5]
        raise RuntimeError(
            f'predicate hubs: unassigned record(s) lack an '
            f'embedding '
            f'— run the component enrichment pass first: {names}'
        )

    top_k = config.get_settings().stages.search.top_k
    gate = llm.gate(max_concurrency)
    new_hubs: dict[str, dict] = {}
    assignments: list[dict] = []
    aliases: dict[str, set[str]] = {}
    records_by_hub: dict[str, list[dict]] = {}
    hierarchy_edges: set[tuple[str, str]] = set()

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
        merged_hub, hierarchy, _, _ = await _choose_hub(
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
            for candidate_uuid, more_general in hierarchy:
                edge = (
                    (merged_hub, candidate_uuid)
                    if more_general == 'left'
                    else (candidate_uuid, merged_hub)
                )
                hierarchy_edges.add(edge)

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

    refreshed_hubs = await _refresh_source_hubs(
        spec,
        records_by_hub,
        set(new_hubs),
        synthesizer,
        gate,
    )
    result = {
        'assigned': len(assignments),
        'new_hubs': len(new_hubs),
        'hierarchies': len(hierarchy_edges),
        'changed_hubs': sorted(aliases),
    }
    result.update(
        assignments=assignments,
        hubs=refreshed_hubs,
    )
    return result


async def align_meta_hubs(
    source_hub_uuids: list[str],
    *,
    session_factory: callable,
    max_concurrency: int | None = None,
    language_model: Any,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Align selected source hubs to meta predicate hubs."""
    spec = PredicateHubSpec(
        graph=writer.PREDICATE_HUB_LABEL,
        stage_name='predicate_hubs',
        hub_id_factory=meta_hub_id_factory,
        source_resolver=lambda _: None,
        adjudication_context='Compare source-local hubs across different sources.',
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_predicate_meta_hubs,
        persist_hubs=writer.persist_predicate_hubs,
        rebuild_names=None,
        rebuild_triplets=None,
    )
    spec.all_source_hubs = queries.all_predicate_source_hubs
    spec.qualified_meta_hub_uuids = queries.qualified_predicate_meta_hub_uuids
    spec.index_name = 'meta_predicate_hub_embedding'
    spec.clear_hubs = writer.clear_predicate_meta_hubs
    spec.persist_hubs = writer.persist_predicate_hubs
    spec.attach_meta_hubs = writer.attach_predicate_meta_hubs
    spec.clear_invalid_meta_hubs = writer.clear_invalid_predicate_meta_hubs
    spec.rebuild_meta_names = lambda *a, **kw: _rebuild_meta_names_callback(
        'predicate', *a, **kw
    )
    spec.rebuild_triplets = _rebuild_triplets_callback

    if not source_hub_uuids:
        return {'aligned': 0, 'new_hubs': 0, 'hierarchies': 0}
    all_records = await queries.all_predicate_source_hubs(session_factory)
    require_meta_sources(spec, all_records)
    records = await queries.all_predicate_source_hubs(
        session_factory, hub_uuids=source_hub_uuids
    )
    if len(records) != len(set(source_hub_uuids)):
        found = {record['uuid'] for record in records}
        missing = sorted(set(source_hub_uuids) - found)
        raise RuntimeError(
            f'predicate hubs (meta): source hub(s) not found: {missing}'
        )
    require_meta_sources(spec, records)
    if any(not record.get('embedding') for record in records):
        missing = [
            record['uuid'] for record in records if not record.get('embedding')
        ]
        raise RuntimeError(
            f'predicate hubs (meta): source hub(s) lack an embedding: {missing}'
        )

    eligible_meta_hubs = await queries.qualified_predicate_meta_hub_uuids(
        session_factory
    )
    top_k = config.get_settings().stages.search.top_k
    gate = llm.gate(max_concurrency)
    new_hubs: dict[str, dict] = {}
    assignments: list[dict] = []
    aliases: dict[str, set[str]] = {}
    hierarchy_edges: set[tuple[str, str]] = set()

    for record in records:
        candidates = [
            {
                **candidate,
                'name': candidate['canonical_name'],
                'aliases': candidate.get('aliases') or [],
                'source': None,
            }
            for candidate in await queries.vector_search(
                session_factory,
                index_name='meta_predicate_hub_embedding',
                query_embedding=record['embedding'],
                top_k=top_k,
            )
        ]
        candidates.extend(
            {
                **hub,
                'name': hub['canonical_name'],
                'aliases': hub.get('aliases') or [],
                'source': None,
                'score': embeddings.cosine_similarity(
                    record['embedding'], hub['embedding']
                ),
            }
            for hub in new_hubs.values()
        )
        candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
        meta_hub, hierarchy, score, decision = await _choose_hub(
            spec,
            record,
            candidates,
            adjudicator,
            gate,
            spec.adjudication_context,
        )
        if meta_hub is None:
            hub = await _new_hub(record, synthesizer, gate, spec)
            meta_hub = hub['uuid']
            existing = new_hubs.setdefault(meta_hub, hub)
            existing['members'] = list(
                dict.fromkeys(existing['members'] + hub['members'])
            )
            existing['aliases'] = sorted(
                set(existing['aliases']) | set(hub['aliases'])
            )
            for candidate_uuid, more_general in hierarchy:
                edge = (
                    (meta_hub, candidate_uuid)
                    if more_general == 'left'
                    else (candidate_uuid, meta_hub)
                )
                hierarchy_edges.add(edge)
            decision = 'Hierarchy' if hierarchy else 'Separate'
        if meta_hub in new_hubs:
            new_hubs[meta_hub]['members'] = list(
                dict.fromkeys(new_hubs[meta_hub]['members'] + [record['uuid']])
            )
            new_hubs[meta_hub]['aliases'] = sorted(
                set(new_hubs[meta_hub]['aliases']) | _aliases(record)
            )
        assignments.append(
            {
                'source_hub': record['uuid'],
                'meta_hub': meta_hub,
                'score': score,
                'decision': decision,
            }
        )
        aliases.setdefault(meta_hub, set()).update(_aliases(record))
        for candidate in candidates:
            if candidate['uuid'] == meta_hub:
                aliases[meta_hub].update(_aliases(candidate))
                break

    source_by_hub = {record['uuid']: record.get('source') for record in records}
    assignment_sources: dict[str, set[str]] = {}
    for assignment in assignments:
        source = source_by_hub.get(assignment['source_hub'])
        if source:
            assignment_sources.setdefault(assignment['meta_hub'], set()).add(
                source
            )
    qualifying_ids = eligible_meta_hubs | {
        hub
        for hub, sources in assignment_sources.items()
        if hub in new_hubs and len(sources) >= 2
    }
    assignments = [a for a in assignments if a['meta_hub'] in qualifying_ids]
    new_hubs = {
        key: value for key, value in new_hubs.items() if key in qualifying_ids
    }
    aliases = {
        key: value for key, value in aliases.items() if key in qualifying_ids
    }
    hierarchy_edges = {
        edge
        for edge in hierarchy_edges
        if edge[0] in qualifying_ids and edge[1] in qualifying_ids
    }
    await writer.persist_predicate_hubs(
        list(new_hubs.values()),
        session_factory=session_factory,
        subsumption_edges=[],
        tier='meta',
    )
    await writer.attach_predicate_meta_hubs(
        assignments,
        aliases=[
            {'hub': hub, 'aliases': sorted(values)}
            for hub, values in aliases.items()
        ],
        subsumption_edges=[
            {'general': general, 'specific': specific}
            for general, specific in sorted(hierarchy_edges)
        ],
        session_factory=session_factory,
    )
    await writer.clear_invalid_predicate_meta_hubs(
        session_factory=session_factory
    )
    await _rebuild_meta_names_callback(
        'predicate',
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    await _rebuild_triplets_callback(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {
        'aligned': len(assignments),
        'new_hubs': len(new_hubs),
        'hierarchies': len(hierarchy_edges),
    }


async def rebuild_meta(
    *,
    session_factory,
    language_model,
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild cross-source predicate hubs from source-local hubs."""
    spec = PredicateHubSpec(
        graph=writer.PREDICATE_HUB_LABEL,
        stage_name='predicate_hubs',
        hub_id_factory=meta_hub_id_factory,
        source_resolver=lambda _: None,
        adjudication_context='Compare source-local hubs across different sources.',
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_predicate_meta_hubs,
        persist_hubs=writer.persist_predicate_hubs,
        rebuild_names=None,
        rebuild_triplets=None,
    )
    spec.all_source_hubs = queries.all_predicate_source_hubs
    spec.qualified_meta_hub_uuids = queries.qualified_predicate_meta_hub_uuids
    spec.index_name = 'meta_predicate_hub_embedding'
    spec.clear_hubs = writer.clear_predicate_meta_hubs
    spec.persist_hubs = writer.persist_predicate_hubs
    spec.attach_meta_hubs = writer.attach_predicate_meta_hubs
    spec.clear_invalid_meta_hubs = writer.clear_invalid_predicate_meta_hubs
    spec.rebuild_meta_names = lambda *a, **kw: _rebuild_meta_names_callback(
        'predicate', *a, **kw
    )
    spec.rebuild_triplets = _rebuild_triplets_callback

    records = await queries.all_predicate_source_hubs(session_factory)
    require_meta_sources(spec, records)
    result = await build_hubs(
        records,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    result = _qualify_meta_result(result, records)
    await writer.clear_predicate_meta_hubs(session_factory=session_factory)
    await writer.persist_predicate_hubs(
        result['hubs'],
        session_factory=session_factory,
        subsumption_edges=result['subsumption_edges'],
        tier='meta',
    )
    await _rebuild_meta_names_callback(
        'predicate',
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    await _rebuild_triplets_callback(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {'meta_hubs': len(result['hubs']), 'source_hubs': result['records']}


async def rebuild(
    source: str,
    *,
    session_factory,
    language_model,
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild source-local predicate hubs from persisted components."""
    spec = PredicateHubSpec(
        graph=writer.PREDICATE_HUB_LABEL,
        stage_name='predicate_hubs',
        hub_id_factory=source_hub_id_factory,
        source_resolver=lambda records: max(
            (r.get('source') for r in records),
            key=lambda s: sum(1 for r in records if r.get('source') == s),
        ),
        adjudication_context='Compare durable component mentions within one source.',
        synthesis_context='Synthesize a source-local semantic hub.',
        all_components=queries.all_predicate_components,
        clear_hubs=writer.clear_predicate_hubs,
        persist_hubs=writer.persist_predicate_hubs,
        rebuild_names=_rebuild_names_callback,
        rebuild_triplets=_rebuild_triplets_callback,
    )
    component_rows = await queries.all_predicate_components(
        session_factory, source
    )
    result = await build_hubs(
        component_rows,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_predicate_hubs(source, session_factory=session_factory)
    await writer.persist_predicate_hubs(
        result['hubs'],
        session_factory=session_factory,
        subsumption_edges=result['subsumption_edges'],
        tier='source',
    )
    await _rebuild_names_callback(
        'predicate',
        source,
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    await _rebuild_triplets_callback(
        language_model=language_model,
        source=source,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {'clusters': result['clusters'], 'records': result['records']}


class PredicateHubNode:
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
        predicate_bundle = models.HubBuildBundle(
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
                for component in construction_bundle.predicate_hub_components
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
                for record in construction_bundle.predicate_hub_records
            ),
        )

        result = await assign_source_hubs(
            predicate_bundle,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        construction_bundle.predicate_hub_assignments = result.get(
            'assignments', []
        )
        construction_bundle.predicate_hub_records = result.get('hubs', [])
        return {
            'predicate_hub_assignments': construction_bundle.predicate_hub_assignments,
            'predicate_hub_records': construction_bundle.predicate_hub_records,
            'construction_bundle': construction_bundle,
        }
