"""Build reusable concept and relation abstractions from source evidence.

Source-level triplet components carry local names and passage-grounded glosses.
This module builds those mentions into reusable EntityHub and PredicateHub
concepts, preserving source provenance while synthesizing
standalone learner-facing descriptions.

The hubs are abstractions, not replacements for the raw evidence: source
triplets remain the factual record from which the hubs are derived.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from kms import config
from kms.construction import name_hubs, triplet_hubs
from kms.core import content, embeddings, llm
from kms.graph import hubs, queries, writer

logger = logging.getLogger(__name__)

HubIdFactory = Callable[[str, list[dict], dict], str]
RecordAdapter = Callable[[list[dict]], list[dict]]
SourceResolver = Callable[[list[dict]], str | None]


@dataclass(frozen=True)
class HubBuildSpec:
    """Defines tier-specific behavior for building semantic hubs."""

    tier: Literal['source', 'meta']
    hub_id_factory: HubIdFactory
    source_resolver: SourceResolver
    record_adapter: RecordAdapter
    adjudication_context: str
    synthesis_context: str


def _coarse_clusters(
    mentions: list[dict], recall_threshold: float
) -> list[list[dict]]:
    """Groups mentions into connected components by cosine similarity.

    An edge joins two mentions whose similarity is at least the recall
    threshold, so components are deliberately generous and leave
    precision decisions to the adjudicator.
    """
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
    """Splits one coarse component into final clusters.

    A pivot is picked per cluster; members are merged when similarity
    is above ``merge_above``, left for a later cluster when below
    ``separate_below``, and adjudicated by the LLM in between.

    Returns:
        ``(clusters, subsumption)`` where subsumption entries are
        ``(general_index, specific_index)`` into the clusters list.
    """
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
        """Synthesizes the definition for one cluster."""
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


def _source_hub_id(kind: str, records: list[dict], definition: dict) -> str:
    """Returns the deterministic id for a source-local hub."""
    identity = '|'.join(sorted(record['uuid'] for record in records))
    return hubs.hub_uuid(kind, _dominant_source(records), identity)


def _meta_hub_id(kind: str, records: list[dict], definition: dict) -> str:
    """Returns an id derived from the stable source-hub membership set."""
    member_ids = sorted(record['uuid'] for record in records)
    if not member_ids:
        raise ValueError('meta hub clusters must contain source hubs')
    return hubs.meta_hub_uuid(kind, '|'.join(member_ids))


def _component_records(rows: list[dict]) -> list[dict]:
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


def _source_hub_records(rows: list[dict]) -> list[dict]:
    return [
        {
            'uuid': row['uuid'],
            'name': row['name'],
            'aliases': row.get('aliases') or [],
            'description': row.get('description'),
            'embedding': row.get('embedding'),
            'source': row.get('source'),
        }
        for row in rows
    ]


def _require_meta_sources(kind: str, records: list[dict]) -> set[str]:
    sources = {
        record.get('source') for record in records if record.get('source')
    }
    if len(sources) < 2:
        raise RuntimeError(
            f'hub_builder (meta/{kind}): requires at least two distinct '
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


def _source_scope(records: list[dict]) -> str:
    return _dominant_source(records)


def _no_source(records: list[dict]) -> None:
    return None


SOURCE_SPEC = HubBuildSpec(
    tier='source',
    hub_id_factory=_source_hub_id,
    source_resolver=_source_scope,
    record_adapter=_component_records,
    adjudication_context=(
        'Compare durable component mentions within one source.'
    ),
    synthesis_context='Synthesize a source-local semantic hub.',
)

META_SPEC = HubBuildSpec(
    tier='meta',
    hub_id_factory=_meta_hub_id,
    source_resolver=_no_source,
    record_adapter=_source_hub_records,
    adjudication_context='Compare source-local hubs across different sources.',
    synthesis_context='Synthesize a cross-source semantic hub.',
)


async def build_hubs(
    kind: str,
    records: list[dict],
    *,
    spec: HubBuildSpec,
    recall_threshold: float | None = None,
    merge_above: float | None = None,
    separate_below: float | None = None,
    max_concurrency: int | None = None,
    adjudicator: Any | None = None,
    synthesizer: Any | None = None,
) -> dict:
    """Builds reusable hubs from already-loaded source records.

    Records must contain ``uuid``, ``name``, ``description``, ``embedding``,
    and optionally ``source`` and ``aliases``. The spec supplies tier-specific
    IDs, source handling, input adapters, and prompt context.

    This function intentionally does not read or write Neo4j. Callers can
    use the same clustering and synthesis engine for source hubs and
    hub-over-hubs meta hubs, then persist the result at the appropriate tier.
    """
    if adjudicator is None or synthesizer is None:
        raise TypeError('adjudicator and synthesizer are required')
    records = spec.record_adapter(records)
    stage = getattr(
        config.get_settings().stages,
        f'{kind}_hubs',
    )
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
            f'hub_builder ({kind}): {len(missing)} record(s) lack an '
            f'embedding — run the component enrichment pass first: {names}'
        )

    components = _coarse_clusters(records, recall_threshold)
    logger.info(
        'hub_builder (%s/%s): %d coarse component(s) at recall %.2f',
        spec.tier,
        kind,
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
        make_hub_id(kind, cluster, definition)
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


def _hub_index(kind: str, tier: str) -> str:
    return (
        f'{tier}_{kind}_hub_embedding'
        if tier == 'meta'
        else (f'{kind}_hub_embedding')
    )


def _candidate_record(candidate: dict, source: str | None) -> dict:
    return {
        **candidate,
        'name': candidate['canonical_name'],
        'aliases': candidate.get('aliases') or [],
        'source': source,
    }


async def _choose_hub(
    kind: str,
    record: dict,
    candidates: list[dict],
    adjudicator: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> tuple[str | None, list[tuple[str, str]], float, str]:
    stage = getattr(config.get_settings().stages, f'{kind}_hubs')
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
    kind: str,
    record: dict,
    synthesizer: Any,
    gate: asyncio.Semaphore,
    spec: HubBuildSpec,
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
        'uuid': spec.hub_id_factory(kind, [record], definition),
        'source': spec.source_resolver([record]),
        'canonical_name': definition['canonical_name'],
        'aliases': sorted(_aliases(record)),
        'description': definition['description'],
        'embedding': vector[0],
        'members': [record['uuid']],
    }


def _aliases(record: dict) -> set[str]:
    return {
        value for value in [record['name'], *record.get('aliases', [])] if value
    }


async def _refresh_source_hubs(
    kind: str,
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
                SOURCE_SPEC.synthesis_context,
            )
        )[0]
        vector = await embeddings.embedder().embed(
            [
                content.Content.from_text(
                    f'{definition["canonical_name"]}: '
                    f'{definition["description"]}'
                )
            ]
        )
        hub = {
            'uuid': hub_uuid,
            'source': SOURCE_SPEC.source_resolver(records),
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


async def assign_source_hubs(
    kind: str,
    source: str,
    *,
    session_factory: Callable,
    max_concurrency: int | None = None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Assign unassigned components to source-local semantic hubs.

    Args:
        kind: The component kind, either ``entity`` or ``predicate``.
        source: The source key whose components should be assigned.
        session_factory: Creates async Neo4j sessions.
        max_concurrency: Optional limit for concurrent model calls.

    Returns:
        Assignment, hub-creation, hierarchy, and changed-hub counts.

    Raises:
        RuntimeError: If a component is missing its required embedding.
    """
    records = await queries.unassigned_components(session_factory, kind, source)
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
            f'hub_builder ({kind}): unassigned record(s) lack an embedding '
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
            _candidate_record(candidate, source)
            for candidate in await queries.vector_search(
                session_factory,
                index_name=_hub_index(kind, 'source'),
                query_embedding=record['embedding'],
                top_k=top_k,
                source=source,
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
            kind,
            record,
            candidates,
            adjudicator,
            gate,
            SOURCE_SPEC.adjudication_context,
        )

        if merged_hub is None:
            hub = await _new_hub(kind, record, synthesizer, gate, SOURCE_SPEC)
            merged_hub = hub['uuid']
            existing = new_hubs.setdefault(merged_hub, hub)
            existing['members'] = list(
                dict.fromkeys(existing['members'] + hub['members'])
            )
            existing['aliases'] = sorted(
                set(existing['aliases']) | set(hub['aliases'])
            )
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
        kind,
        records_by_hub,
        set(new_hubs),
        synthesizer,
        gate,
    )
    await writer.persist_hubs(
        kind,
        refreshed_hubs,
        session_factory=session_factory,
        subsumption_edges=[
            {'general': general, 'specific': specific}
            for general, specific in sorted(hierarchy_edges)
        ],
        tier='source',
    )
    await writer.attach_source_components(
        kind,
        assignments,
        aliases=[
            {'hub': hub, 'aliases': sorted(values)}
            for hub, values in aliases.items()
        ],
        session_factory=session_factory,
    )
    return {
        'assigned': len(assignments),
        'new_hubs': len(new_hubs),
        'hierarchies': len(hierarchy_edges),
        'changed_hubs': sorted(aliases),
    }


async def align_meta_hubs(
    kind: str,
    source_hub_uuids: list[str],
    *,
    session_factory: Callable,
    max_concurrency: int | None = None,
    language_model: Any,
    adjudicator: Any | None = None,
    synthesizer: Any | None = None,
) -> dict:
    """Align selected source-local hubs to qualified meta hubs.

    Args:
        kind: The hub kind, either ``entity`` or ``predicate``.
        source_hub_uuids: Source-local hub UUIDs to align.
        language_model: The model used for adjudication and synthesis.
        session_factory: Creates async Neo4j sessions.
        max_concurrency: Optional limit for concurrent model calls.

    Returns:
        Alignment, hub-creation, and hierarchy counts.

    Raises:
        RuntimeError: If source hubs are missing or lack embeddings.
    """
    if not source_hub_uuids:
        return {'aligned': 0, 'new_hubs': 0, 'hierarchies': 0}
    all_records = await queries.all_source_hubs(session_factory, kind)
    _require_meta_sources(kind, all_records)
    records = await queries.all_source_hubs(
        session_factory, kind, hub_uuids=source_hub_uuids
    )
    if len(records) != len(set(source_hub_uuids)):
        found = {record['uuid'] for record in records}
        missing = sorted(set(source_hub_uuids) - found)
        raise RuntimeError(
            f'hub_builder (meta/{kind}): source hub(s) not found: {missing}'
        )
    _require_meta_sources(kind, records)
    if any(not record.get('embedding') for record in records):
        missing = [
            record['uuid'] for record in records if not record.get('embedding')
        ]
        raise RuntimeError(
            f'hub_builder (meta/{kind}): source hub(s) lack an embedding: '
            f'{missing}'
        )

    eligible_meta_hubs = await queries.qualified_meta_hub_uuids(
        session_factory, kind
    )
    top_k = config.get_settings().stages.search.top_k
    gate = llm.gate(max_concurrency)
    new_hubs: dict[str, dict] = {}
    assignments: list[dict] = []
    aliases: dict[str, set[str]] = {}
    hierarchy_edges: set[tuple[str, str]] = set()

    for record in records:
        candidates = [
            _candidate_record(candidate, None)
            for candidate in await queries.vector_search(
                session_factory,
                index_name=_hub_index(kind, 'meta'),
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
            kind,
            record,
            candidates,
            adjudicator,
            gate,
            META_SPEC.adjudication_context,
        )
        if meta_hub is None:
            hub = await _new_hub(kind, record, synthesizer, gate, META_SPEC)
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
        aliases.setdefault(meta_hub, set())
        aliases[meta_hub].update(_aliases(record))
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
        meta_hub
        for meta_hub, sources in assignment_sources.items()
        if meta_hub in new_hubs and len(sources) >= 2
    }
    assignments = [
        assignment
        for assignment in assignments
        if assignment['meta_hub'] in qualifying_ids
    ]
    new_hubs = {
        hub_uuid: hub
        for hub_uuid, hub in new_hubs.items()
        if hub_uuid in qualifying_ids
    }
    aliases = {
        hub_uuid: values
        for hub_uuid, values in aliases.items()
        if hub_uuid in qualifying_ids
    }
    hierarchy_edges = {
        edge
        for edge in hierarchy_edges
        if edge[0] in qualifying_ids and edge[1] in qualifying_ids
    }

    await writer.persist_hubs(
        kind,
        list(new_hubs.values()),
        session_factory=session_factory,
        subsumption_edges=[],
        tier='meta',
    )
    await writer.attach_meta_hubs(
        kind,
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
    await writer.clear_invalid_meta_hubs(
        kind,
        session_factory=session_factory,
    )
    await name_hubs.rebuild_meta(
        kind,
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    await triplet_hubs.rebuild_meta(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {
        'aligned': len(assignments),
        'new_hubs': len(new_hubs),
        'hierarchies': len(hierarchy_edges),
    }


async def rebuild_meta_hubs(
    kind: str,
    *,
    session_factory: Callable,
    language_model: Any,
    adjudicator: Any,
    synthesizer: Any,
    recall_threshold: float | None = None,
    merge_above: float | None = None,
    separate_below: float | None = None,
    max_concurrency: int | None = None,
) -> dict:
    """Explicitly rebuilds one disposable meta-hub kind.

    Meta rebuilding is a maintenance operation over persisted source-local
    hubs and is not part of normal ingestion.
    """
    source_hub_rows = await queries.all_source_hubs(session_factory, kind)
    records = source_hub_rows
    _require_meta_sources(kind, records)
    logger.info(
        'hub_builder (meta/%s): %d source hub(s) read',
        kind,
        len(records),
    )
    result = await build_hubs(
        kind,
        records,
        recall_threshold=recall_threshold,
        merge_above=merge_above,
        separate_below=separate_below,
        max_concurrency=max_concurrency,
        spec=META_SPEC,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    result = _qualify_meta_result(result, records)
    await writer.clear_meta_hubs(kind, session_factory=session_factory)
    await writer.persist_hubs(
        kind,
        result['hubs'],
        session_factory=session_factory,
        subsumption_edges=result['subsumption_edges'],
        tier='meta',
    )
    await name_hubs.rebuild_meta(
        kind,
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    await triplet_hubs.rebuild_meta(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    logger.info(
        'hub_builder (meta/%s): %d hub(s) persisted',
        kind,
        len(result['hubs']),
    )
    return {
        'clusters': result['clusters'],
        'hubs': len(result['hubs']),
    }


async def rebuild_hubs(
    kind: str,
    *,
    session_factory: Callable,
    source: str,
    language_model: Any,
    recall_threshold: float | None = None,
    merge_above: float | None = None,
    separate_below: float | None = None,
    max_concurrency: int | None = None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Explicitly rebuilds source-local hubs for one source and kind.

    Normal ingestion uses :func:`assign_source_hubs` so unrelated sources
    remain untouched.
    """
    component_rows = await queries.all_components(
        session_factory, kind, source=source
    )
    logger.info(
        'hub_builder (source/%s, %s): %d component(s) read',
        kind,
        source,
        len(component_rows),
    )
    result = await build_hubs(
        kind,
        component_rows,
        recall_threshold=recall_threshold,
        merge_above=merge_above,
        separate_below=separate_below,
        max_concurrency=max_concurrency,
        spec=SOURCE_SPEC,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_source_hubs(
        kind,
        source,
        session_factory=session_factory,
    )
    await writer.persist_hubs(
        kind,
        result['hubs'],
        session_factory=session_factory,
        subsumption_edges=result['subsumption_edges'],
        tier='source',
    )
    await name_hubs.rebuild(
        kind,
        source,
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    await triplet_hubs.rebuild(
        language_model=language_model,
        session_factory=session_factory,
        source=source,
        max_concurrency=max_concurrency,
    )
    logger.info(
        'hub_builder (source/%s, %s): %d hub(s) persisted',
        kind,
        source,
        len(result['hubs']),
    )
    return {
        'clusters': result['clusters'],
        'records': result['records'],
    }
