"""Build cross-source global predicate hubs from persisted local hubs."""

import asyncio
from dataclasses import dataclass
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import clustering, embeddings, llm, models, module
from kms.graph import local_predicate_hubs as graph_predicate_hubs
from kms.graph import queries, writer

from . import name_hubs, triplet_hubs


@dataclass
class GlobalPredicateHubSpec:
    """Configuration shared by global predicate-hub operations."""

    graph: Any
    stage_name: str
    hub_id_factory: Any
    source_resolver: Any
    adjudication_context: str
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


class PredicateHubDefinition(BaseModel):
    canonical_name: str = Field(
        description='The canonical name for the predicate relation.'
    )
    description: str = Field(
        description='A standalone canonical predicate description supported by the supplied evidence.'
    )


class PredicateHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one source-local canonical predicate concept from supplied
    relation phrases and their passage-grounded descriptions. Generalize only
    what the evidence supports. Preserve the supported meaning and do not
    mention the source or invent facts.
    """

    request: models.HubSynthesisInput = dspy.InputField()
    result: PredicateHubDefinition = dspy.OutputField()


class PredicateHubAdjudicationSignature(dspy.Signature):
    r"""
    Compare two predicate mentions. Return TRUE only for the same canonical
    relation. Return FALSE for broader, narrower, inverse, or merely related
    relations.
    """

    comparison: models.HubMentionComparisonInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


class PredicateHubSynthesizer(module.Module):
    signature = PredicateHubSynthesisSignature
    record_name = 'predicate_hub_synthesizer'

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
        result = PredicateHubDefinition.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


class PredicateHubAdjudicator(module.Module):
    signature = PredicateHubAdjudicationSignature
    record_name = 'predicate_hub_adjudicator'

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


async def _synthesize_global_definitions(
    clusters: list[list[dict]],
    synthesizer: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> list[dict]:
    async def synthesize(cluster: list[dict]) -> dict:
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
                surface_forms=forms, descriptions=descriptions, scope=scope
            )
        return {'canonical_name': name, 'description': description}

    return list(
        await asyncio.gather(*(synthesize(cluster) for cluster in clusters))
    )


def _global_coarse_clusters(
    mentions: list[dict], recall_threshold: float
) -> list[list[dict]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(mentions))}
    for i in range(len(mentions)):
        for j in range(i + 1, len(mentions)):
            if (
                embeddings.cosine_similarity(
                    mentions[i]['embedding'], mentions[j]['embedding']
                )
                >= recall_threshold
            ):
                adjacency[i].append(j)
                adjacency[j].append(i)
    visited: set[int] = set()
    clusters = []
    for start in range(len(mentions)):
        if start in visited:
            continue
        component = []
        stack = [start]
        while stack:
            index = stack.pop()
            if index in visited:
                continue
            visited.add(index)
            component.append(mentions[index])
            stack.extend(adjacency[index])
        clusters.append(component)
    return clusters


def _global_central_mention(component: list[dict]) -> dict:
    if len(component) == 1:
        return component[0]
    return max(
        component,
        key=lambda candidate: sum(
            embeddings.cosine_similarity(
                candidate['embedding'], other['embedding']
            )
            for other in component
            if other is not candidate
        ),
    )


def _global_mention_text(record: dict) -> str:
    mention = models.HubMentionInput.from_record(record)
    return ' '.join([mention.name, *mention.aliases, mention.description or ''])


async def _global_adjudicate_component(
    component: list[dict],
    adjudicator: Any,
    stage: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> list[list[dict]]:
    unassigned = list(component)
    clusters = []
    while unassigned:
        pivot = _global_central_mention(unassigned)
        unassigned = [member for member in unassigned if member is not pivot]
        selected = (
            await clustering.select_reranked_candidates(
                _global_mention_text(pivot),
                unassigned,
                _global_mention_text,
                top_n=stage.rerank_top_n,
            )
            if stage.rerank_top_n
            else unassigned
        )
        selected_ids = {id(member) for member in selected}
        remaining = [
            member for member in unassigned if id(member) not in selected_ids
        ]
        cluster = [pivot]
        boundary = []
        for member in selected:
            score = embeddings.cosine_similarity(
                pivot['embedding'], member['embedding']
            )
            if score >= stage.merge_above:
                cluster.append(member)
            elif score > stage.separate_below:
                boundary.append(member)
            else:
                remaining.append(member)

        async def decide(member: dict, pivot_record: dict = pivot) -> Any:
            async with gate:
                return await adjudicator.aforward(
                    left=models.HubMentionInput.from_record(pivot_record),
                    right=models.HubMentionInput.from_record(member),
                    scope=scope,
                )

        decisions = await asyncio.gather(
            *(decide(member) for member in boundary)
        )
        for member, decision in zip(boundary, decisions, strict=True):
            (cluster if decision else remaining).append(member)
        clusters.append(cluster)
        unassigned = remaining
    return clusters


async def _build_global_predicate_hubs(
    records: list[dict],
    *,
    spec: GlobalPredicateHubSpec,
    max_concurrency: int | None,
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    if not records:
        return {'clusters': 0, 'records': 0, 'hubs': []}
    stage = getattr(config.get_settings().stages, spec.stage_name)
    if any(not record.get('embedding') for record in records):
        raise RuntimeError('predicate hubs: records must have embeddings')
    components = _global_coarse_clusters(records, stage.recall_threshold)
    gate = llm.gate(max_concurrency)
    clusters = []
    for component in components:
        clusters.extend(
            await _global_adjudicate_component(
                component, adjudicator, stage, gate, spec.adjudication_context
            )
        )
    definitions = await _synthesize_global_definitions(
        clusters, synthesizer, gate, spec.synthesis_context
    )
    vectors = await embeddings.embedder().embed(
        [f'{d["canonical_name"]}: {d["description"]}' for d in definitions]
    )
    cluster_ids = [
        spec.hub_id_factory(cluster, definition)
        for cluster, definition in zip(clusters, definitions, strict=True)
    ]
    hubs = []
    for cluster, definition, vector, hub_uuid in zip(
        clusters, definitions, vectors, cluster_ids, strict=True
    ):
        members = [member['uuid'] for member in cluster]
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
    return {'clusters': len(clusters), 'records': len(records), 'hubs': hubs}


async def _choose_hub(
    spec: GlobalPredicateHubSpec,
    record: dict,
    candidates: list[dict],
    adjudicator: Any,
    gate: asyncio.Semaphore,
    scope: str,
) -> tuple[str | None, list[tuple[str, str]], float, str]:
    """Choose a matching global hub using score and adjudication thresholds."""
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
    spec: GlobalPredicateHubSpec,
) -> dict:
    """Synthesize a new global hub for an unmatched source hub."""
    surface_forms = list(
        dict.fromkeys(
            form
            for form in [record['name'], *record.get('aliases', [])]
            if form
        )
    )
    descriptions = sorted(
        {
            member['description']
            for member in [record]
            if member.get('description')
        }
    )
    async with gate:
        canonical_name, description = await synthesizer.aforward(
            surface_forms=surface_forms,
            descriptions=descriptions,
            scope=spec.synthesis_context,
        )
    definition = {
        'canonical_name': canonical_name,
        'description': description,
    }
    vector = await embeddings.embedder().embed(
        [f'{canonical_name}: {description}']
    )
    return {
        'uuid': spec.hub_id_factory([record], definition),
        'source': spec.source_resolver([record]),
        'canonical_name': canonical_name,
        'aliases': sorted(
            {
                value
                for value in [record.get('name'), *record.get('aliases', [])]
                if value
            }
        ),
        'description': description,
        'embedding': vector[0],
        'members': [record['uuid']],
    }


def _aliases(record: dict) -> set[str]:
    name = record.get('name') or record.get('canonical_name')
    return {value for value in [name, *record.get('aliases', [])] if value}


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
    )


def global_hub_id_factory(graph: Any) -> callable:
    """Creates a deterministic meta-hub id function for predicate hubs."""

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


def require_global_sources(
    spec: GlobalPredicateHubSpec, records: list[dict]
) -> set[str]:
    sources = {record['source'] for record in records if record.get('source')}
    if len(sources) < 2:
        raise RuntimeError(
            f'predicate hubs (meta): requires at least two distinct '
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
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Align selected source hubs to meta predicate hubs."""
    spec = GlobalPredicateHubSpec(
        graph=graph_predicate_hubs,
        stage_name='predicate_hubs',
        hub_id_factory=global_hub_id_factory(graph_predicate_hubs),
        source_resolver=lambda _: None,
        adjudication_context='Compare source-local hubs across different sources.',
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_predicate_global_hubs,
        persist_hubs=writer.persist_predicate_hubs,
        rebuild_names=None,
        rebuild_triplets=_rebuild_triplets_callback,
        all_source_hubs=queries.all_predicate_source_hubs,
        qualified_global_hub_uuids=queries.qualified_predicate_meta_hub_uuids,
        index_name='global_predicate_hub_embedding',
        attach_global_hubs=writer.attach_predicate_global_hubs,
        clear_invalid_global_hubs=writer.clear_invalid_predicate_global_hubs,
        rebuild_global_names=lambda *a, **kw: _rebuild_global_names_callback(
            'predicate', *a, **kw
        ),
    )

    if not source_hub_uuids:
        return {'aligned': 0, 'new_hubs': 0}
    all_records = await queries.all_predicate_source_hubs(session_factory)
    require_global_sources(spec, all_records)
    records = await queries.all_predicate_source_hubs(
        session_factory, hub_uuids=source_hub_uuids
    )
    if len(records) != len(set(source_hub_uuids)):
        found = {record['uuid'] for record in records}
        missing = sorted(set(source_hub_uuids) - found)
        raise RuntimeError(
            f'predicate hubs (meta): source hub(s) not found: {missing}'
        )
    require_global_sources(spec, records)
    if any(not record.get('embedding') for record in records):
        missing = [
            record['uuid'] for record in records if not record.get('embedding')
        ]
        raise RuntimeError(
            f'predicate hubs (meta): source hub(s) lack an embedding: {missing}'
        )

    eligible_global_hubs = await queries.qualified_predicate_meta_hub_uuids(
        session_factory
    )
    top_k = config.get_settings().stages.search.top_k
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
                index_name='global_predicate_hub_embedding',
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
        global_hub, _, score, decision = await _choose_hub(
            spec,
            record,
            candidates,
            adjudicator,
            gate,
            spec.adjudication_context,
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
    await writer.persist_predicate_hubs(
        list(new_hubs.values()),
        session_factory=session_factory,
        tier='meta',
    )
    await writer.attach_predicate_global_hubs(
        assignments,
        aliases=[
            {'hub': hub, 'aliases': sorted(values)}
            for hub, values in aliases.items()
        ],
        session_factory=session_factory,
    )
    await writer.clear_invalid_predicate_global_hubs(
        session_factory=session_factory
    )
    await _rebuild_global_names_callback(
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
    }


async def rebuild_global(
    *,
    session_factory,
    language_model,
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild cross-source predicate hubs from source-local hubs."""
    spec = GlobalPredicateHubSpec(
        graph=graph_predicate_hubs,
        stage_name='predicate_hubs',
        hub_id_factory=global_hub_id_factory(graph_predicate_hubs),
        source_resolver=lambda _: None,
        adjudication_context='Compare source-local hubs across different sources.',
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_predicate_global_hubs,
        persist_hubs=writer.persist_predicate_hubs,
        rebuild_names=None,
        rebuild_triplets=_rebuild_triplets_callback,
        all_source_hubs=queries.all_predicate_source_hubs,
        qualified_global_hub_uuids=queries.qualified_predicate_meta_hub_uuids,
        index_name='global_predicate_hub_embedding',
        attach_global_hubs=writer.attach_predicate_global_hubs,
        clear_invalid_global_hubs=writer.clear_invalid_predicate_global_hubs,
        rebuild_global_names=lambda *a, **kw: _rebuild_global_names_callback(
            'predicate', *a, **kw
        ),
    )
    records = await queries.all_predicate_source_hubs(session_factory)
    result = await _build_global_predicate_hubs(
        records,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_predicate_global_hubs(session_factory=session_factory)
    await writer.persist_predicate_hubs(
        result['hubs'],
        session_factory=session_factory,
        tier='meta',
    )
    await _rebuild_global_names_callback(
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
    return {'global_hubs': len(result['hubs']), 'local_hubs': result['records']}
