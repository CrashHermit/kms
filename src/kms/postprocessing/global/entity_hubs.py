"""Build cross-source global entity hubs from persisted local hubs."""

import asyncio
from dataclasses import dataclass
from typing import Any

from kms import config
from kms.core import batching, context_window, embeddings, llm, models, reranker
from kms.graph import local_entity_hubs as graph_entity_hubs
from kms.graph import queries, writer

from . import name_hubs, triplet_hubs


@dataclass
class GlobalEntityHubSpec:
    """Configuration shared by global entity-hub operations."""

    graph: Any
    stage_name: str
    hub_id_factory: Any
    source_resolver: Any
    synthesis_context: str
    all_components: Any
    clear_hubs: Any
    persist_hubs: Any
    rebuild_names: Any
    rebuild_triplets: Any
    all_source_hubs: Any
    qualified_global_hub_uuids: Any
    index_name: str
    attach_global_hubs: Any
    clear_invalid_global_hubs: Any
    rebuild_global_names: Any

    def __post_init__(self) -> None:
        """Validate the global operation's required callbacks."""
        if self.all_components is not None:
            raise ValueError('global entity specs cannot load local components')


async def _rebuild_global_names_callback(
    domain: str, *, language_model, session_factory, max_concurrency=None
):
    return await name_hubs.rebuild_global(
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
    return await triplet_hubs.rebuild_global(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )


def global_hub_id_factory(graph: Any) -> callable:
    """Creates a deterministic meta-hub id function for entity hubs."""

    def make_id(records: list[dict], definition: dict) -> str:
        member_ids = sorted(record['uuid'] for record in records)
        if not member_ids:
            raise ValueError('meta hub clusters must contain source hubs')
        return graph.global_hub_uuid('|'.join(member_ids))

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


def local_hub_records(rows: list[dict]) -> list[dict]:
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


def _mention_text(record: dict) -> str:
    mention = models.HubMentionInput.from_record(record)
    parts = [mention.name, *mention.aliases]
    if mention.description:
        parts.append(mention.description)
    return ' '.join(parts)


def _aliases(record: dict) -> set[str]:
    name = record.get('name') or record.get('canonical_name')
    return {value for value in [name, *record.get('aliases', [])] if value}


def _choose_entity_hub(
    spec: GlobalEntityHubSpec, candidates: list[dict]
) -> tuple[str | None, list[tuple[str, str]], float, str]:
    stage = getattr(config.get_settings().stages, spec.stage_name)
    for candidate in candidates:
        score = candidate['score']
        if score >= stage.merge_above:
            return candidate['uuid'], [], score, 'Merge'
        if score > stage.separate_below:
            continue
    return None, [], 0.0, 'Separate'


async def _new_hub(
    record: dict,
    synthesizer: Any,
    gate: asyncio.Semaphore,
    spec: GlobalEntityHubSpec,
) -> dict:
    surface_forms = list(
        dict.fromkeys([record['name'], *record.get('aliases', [])])
    )
    descriptions = [record['description']] if record.get('description') else []
    async with gate:
        canonical_name, description = await synthesizer.aforward(
            surface_forms=surface_forms,
            descriptions=descriptions,
            scope=spec.synthesis_context,
        )
    definition = {'canonical_name': canonical_name, 'description': description}
    vector = await embeddings.embedder().embed(
        [f'{canonical_name}: {description}']
    )
    return {
        'uuid': spec.hub_id_factory([record], definition),
        'source': spec.source_resolver([record]),
        'canonical_name': canonical_name,
        'aliases': sorted(_aliases(record)),
        'description': description,
        'embedding': vector[0],
        'members': [record['uuid']],
    }


async def _build_global_entity_hubs(
    records: list[dict],
    *,
    spec: GlobalEntityHubSpec,
    max_concurrency: int | None,
    synthesizer: Any,
) -> dict:
    """Cluster global entity source hubs without construction dependencies."""
    diagnostics = {
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
    diagnostics['records'] = diagnostics['remaining_records'] = len(records)
    if not records:
        return {
            'assignments': [],
            'clusters': 0,
            'diagnostics': diagnostics,
            'hubs': [],
            'records': 0,
        }
    stage = getattr(config.get_settings().stages, spec.stage_name)
    unassigned = sorted(
        (dict(record) for record in records), key=lambda r: r['uuid']
    )
    clusters: list[list[dict]] = []
    while unassigned:
        pivot = unassigned.pop(0)
        automatic: list[dict] = []
        ambiguous: list[tuple[float, dict]] = []
        for candidate in unassigned:
            diagnostics['embedding_comparisons'] += 1
            score = embeddings.cosine_similarity(
                pivot['embedding'], candidate['embedding']
            )
            if score >= stage.merge_above:
                automatic.append(candidate)
            elif score > stage.separate_below:
                ambiguous.append((score, candidate))
        ambiguous.sort(key=lambda item: (-item[0], item[1]['uuid']))
        diagnostics['ambiguous_candidates'] += len(ambiguous)
        admitted = [
            candidate
            for _, candidate in ambiguous[
                : getattr(stage, 'rerank_candidate_limit', 32)
            ]
        ]
        selected: list[dict] = []
        if admitted and reranker.is_configured():
            pivot_text = _mention_text(pivot)
            pivot_tokens = context_window.estimate_text_tokens(pivot_text)
            if pivot_tokens >= stage.comparison_token_budget:
                raise ValueError(
                    'entity hubs: pivot exceeds reranker token budget'
                )
            texts = [
                (candidate, _mention_text(candidate)) for candidate in admitted
            ]
            for _candidate, text in texts:
                if (
                    pivot_tokens + context_window.estimate_text_tokens(text)
                    > stage.comparison_token_budget
                ):
                    raise ValueError(
                        'entity hubs: candidate exceeds reranker token budget'
                    )
            client = reranker.reranker()
            for batch in batching.token_batches(
                texts,
                token_cost=lambda item: context_window.estimate_text_tokens(
                    item[1]
                ),
                token_budget=stage.comparison_token_budget - pivot_tokens,
            ):
                diagnostics['reranker_requests'] += 1
                diagnostics['reranker_candidates'] += len(batch)
                results = await client.rerank(
                    pivot_text,
                    [text for _, text in batch],
                    top_n=stage.rerank_top_n,
                )
                for result in results:
                    if 0 <= result['index'] < len(batch):
                        diagnostics['reranker_selected'] += 1
                        selected.append(batch[result['index']][0])
        cluster = [pivot, *automatic, *selected]
        member_ids = {member['uuid'] for member in cluster}
        unassigned = [
            record for record in unassigned if record['uuid'] not in member_ids
        ]
        clusters.append(cluster)
        diagnostics['locked_groups'] += 1
        diagnostics['remaining_records'] = len(unassigned)
    gate = llm.gate(max_concurrency)
    diagnostics['synthesizer_calls'] = len(clusters)
    definitions = []
    for cluster in clusters:
        forms = list(
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
            name, description = await synthesizer.aforward(
                surface_forms=forms,
                descriptions=descriptions,
                scope=spec.synthesis_context,
            )
        definitions.append({'canonical_name': name, 'description': description})
    vectors = await embeddings.embedder().embed(
        [f'{d["canonical_name"]}: {d["description"]}' for d in definitions]
    )
    hubs = []
    assignments = []
    for cluster, definition, vector in zip(
        clusters, definitions, vectors, strict=True
    ):
        hub_uuid = spec.hub_id_factory(cluster, definition)
        members = sorted(member['uuid'] for member in cluster)
        hubs.append(
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
                'embedding': vector,
                'members': members,
            }
        )
        assignments.extend(
            {'component': member, 'hub': hub_uuid} for member in members
        )
    diagnostics['final_hubs'] = len(hubs)
    return {
        'assignments': sorted(assignments, key=lambda item: item['component']),
        'clusters': len(clusters),
        'diagnostics': diagnostics,
        'hubs': hubs,
        'records': len(records),
    }


def require_global_sources(
    spec: GlobalEntityHubSpec, records: list[dict]
) -> set[str]:
    sources = {record['source'] for record in records if record.get('source')}
    if len(sources) < 2:
        raise RuntimeError(
            f'entity hubs (meta): requires at least two distinct '
            f'sources, found {len(sources)}'
        )
    return sources


def _qualify_global_result(result: dict, records: list[dict]) -> dict:
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
    }


async def align_global_hubs(
    source_hub_uuids: list[str],
    *,
    session_factory: callable,
    max_concurrency: int | None = None,
    language_model: Any,
    synthesizer: Any,
) -> dict:
    """Align selected source hubs to meta entity hubs."""
    spec = GlobalEntityHubSpec(
        graph=graph_entity_hubs,
        stage_name='entity_hubs',
        hub_id_factory=global_hub_id_factory(graph_entity_hubs),
        source_resolver=lambda _: None,
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_entity_global_hubs,
        persist_hubs=writer.persist_entity_hubs,
        rebuild_names=None,
        rebuild_triplets=_rebuild_triplets_callback,
        all_source_hubs=queries.all_entity_source_hubs,
        qualified_global_hub_uuids=queries.qualified_entity_meta_hub_uuids,
        index_name='global_entity_hub_embedding',
        attach_global_hubs=writer.attach_entity_global_hubs,
        clear_invalid_global_hubs=writer.clear_invalid_entity_global_hubs,
        rebuild_global_names=lambda *a, **kw: _rebuild_global_names_callback(
            'entity', *a, **kw
        ),
    )

    if not source_hub_uuids:
        return {'aligned': 0, 'new_hubs': 0}
    all_records = await queries.all_entity_source_hubs(session_factory)
    require_global_sources(spec, all_records)
    records = await queries.all_entity_source_hubs(
        session_factory, hub_uuids=source_hub_uuids
    )
    if len(records) != len(set(source_hub_uuids)):
        found = {record['uuid'] for record in records}
        missing = sorted(set(source_hub_uuids) - found)
        raise RuntimeError(
            f'entity hubs (meta): source hub(s) not found: {missing}'
        )
    require_global_sources(spec, records)
    if any(not record.get('embedding') for record in records):
        missing = [
            record['uuid'] for record in records if not record.get('embedding')
        ]
        raise RuntimeError(
            f'entity hubs (meta): source hub(s) lack an embedding: {missing}'
        )

    eligible_global_hubs = await queries.qualified_entity_meta_hub_uuids(
        session_factory
    )
    top_k = config.get_settings().stages.search.top_k
    stage = config.get_settings().stages.entity_hubs
    gate = llm.gate(max_concurrency)
    new_hubs: dict[str, dict] = {}
    assignments: list[dict] = []
    aliases: dict[str, set[str]] = {}

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
                index_name='global_entity_hub_embedding',
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
        if stage.rerank_top_n and reranker.is_configured():
            selected = await reranker.reranker().rerank(
                _mention_text(record),
                [_mention_text(candidate) for candidate in candidates],
                top_n=stage.rerank_top_n,
            )
            candidates = [
                candidates[result['index']]
                for result in selected
                if 0 <= result['index'] < len(candidates)
            ]
        global_hub, _, score, decision = _choose_entity_hub(
            spec,
            candidates,
        )
        if global_hub is None:
            hub = await _new_hub(record, synthesizer, gate, spec)
            global_hub = hub['uuid']
            existing = new_hubs.setdefault(global_hub, hub)
            existing['members'] = list(
                dict.fromkeys(existing['members'] + hub['members'])
            )
            existing['aliases'] = sorted(
                set(existing['aliases']) | set(hub['aliases'])
            )
        if global_hub in new_hubs:
            new_hubs[global_hub]['members'] = list(
                dict.fromkeys(
                    new_hubs[global_hub]['members'] + [record['uuid']]
                )
            )
            new_hubs[global_hub]['aliases'] = sorted(
                set(new_hubs[global_hub]['aliases']) | _aliases(record)
            )
        assignments.append(
            {
                'source_hub': record['uuid'],
                'meta_hub': global_hub,
                'score': score,
                'decision': decision,
            }
        )
        aliases.setdefault(global_hub, set()).update(_aliases(record))
        for candidate in candidates:
            if candidate['uuid'] == global_hub:
                aliases[global_hub].update(_aliases(candidate))
                break

    source_by_hub = {record['uuid']: record.get('source') for record in records}
    assignment_sources: dict[str, set[str]] = {}
    for assignment in assignments:
        source = source_by_hub.get(assignment['source_hub'])
        if source:
            assignment_sources.setdefault(assignment['meta_hub'], set()).add(
                source
            )
    qualifying_ids = eligible_global_hubs | {
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
    await writer.persist_entity_hubs(
        list(new_hubs.values()),
        session_factory=session_factory,
        tier='meta',
    )
    await writer.attach_entity_global_hubs(
        assignments,
        aliases=[
            {'hub': hub, 'aliases': sorted(values)}
            for hub, values in aliases.items()
        ],
        session_factory=session_factory,
    )
    await _rebuild_triplets_callback(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {
        'aligned': len(assignments),
        'new_hubs': len(new_hubs),
    }


async def rebuild_global(
    *,
    session_factory,
    language_model,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild cross-source entity hubs from source-local hubs."""
    spec = GlobalEntityHubSpec(
        graph=graph_entity_hubs,
        stage_name='entity_hubs',
        hub_id_factory=global_hub_id_factory(graph_entity_hubs),
        source_resolver=lambda _: None,
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_entity_global_hubs,
        persist_hubs=writer.persist_entity_hubs,
        rebuild_names=None,
        rebuild_triplets=_rebuild_triplets_callback,
        all_source_hubs=queries.all_entity_source_hubs,
        qualified_global_hub_uuids=queries.qualified_entity_meta_hub_uuids,
        index_name='global_entity_hub_embedding',
        attach_global_hubs=writer.attach_entity_global_hubs,
        clear_invalid_global_hubs=writer.clear_invalid_entity_global_hubs,
        rebuild_global_names=lambda *a, **kw: _rebuild_global_names_callback(
            'entity', *a, **kw
        ),
    )
    records = await queries.all_entity_source_hubs(session_factory)
    result = await _build_global_entity_hubs(
        records,
        spec=spec,
        max_concurrency=max_concurrency,
        synthesizer=synthesizer,
    )
    result = _qualify_global_result(result, records)
    await writer.clear_entity_global_hubs(session_factory=session_factory)
    await writer.persist_entity_hubs(
        result['hubs'],
        session_factory=session_factory,
        tier='meta',
    )
    await _rebuild_global_names_callback(
        'entity',
        language_model=language_model,
        session_factory=session_factory,
    )
    await _rebuild_triplets_callback(
        language_model=language_model,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    return {'global_hubs': len(result['hubs']), 'local_hubs': result['records']}
