"""Build cross-source global entity hubs from persisted local hubs."""

from dataclasses import dataclass
from typing import Any

from kms import config
from kms.construction import local_entity_hubs as local_hubs
from kms.core import embeddings, llm
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

    def __post_init__(self) -> None:
        """Validate the global operation's required callbacks."""
        if self.all_components is not None:
            raise ValueError('global entity specs cannot load local components')

 
EntityHubAdjudicator = local_hubs.EntityHubAdjudicator
EntityHubSynthesizer = local_hubs.EntityHubSynthesizer
build_hubs = local_hubs.build_hubs
_choose_hub = local_hubs._choose_hub
_new_hub = local_hubs._new_hub
_aliases = local_hubs._aliases

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
    adjudicator: Any,
    synthesizer: Any,
) -> dict:
    """Align selected source hubs to meta entity hubs."""
    spec = GlobalEntityHubSpec(
        graph=graph_entity_hubs,
        stage_name='entity_hubs',
        hub_id_factory=global_hub_id_factory(graph_entity_hubs),
        source_resolver=lambda _: None,
        adjudication_context='Compare source-local hubs across different sources.',
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_entity_global_hubs,
        persist_hubs=writer.persist_entity_hubs,
        rebuild_names=None,
        rebuild_triplets=_rebuild_triplets_callback,
        all_source_hubs=queries.all_entity_source_hubs,
        qualified_global_hub_uuids=queries.qualified_entity_global_hub_uuids,
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

    eligible_global_hubs = await queries.qualified_entity_global_hub_uuids(
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
                dict.fromkeys(new_hubs[global_hub]['members'] + [record['uuid']])
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
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild cross-source entity hubs from source-local hubs."""
    spec = GlobalEntityHubSpec(
        graph=graph_entity_hubs,
        stage_name='entity_hubs',
        hub_id_factory=global_hub_id_factory(graph_entity_hubs),
        source_resolver=lambda _: None,
        adjudication_context='Compare source-local hubs across different sources.',
        synthesis_context='Synthesize a cross-source semantic hub.',
        all_components=None,
        clear_hubs=writer.clear_entity_global_hubs,
        persist_hubs=writer.persist_entity_hubs,
        rebuild_names=None,
        rebuild_triplets=_rebuild_triplets_callback,
        all_source_hubs=queries.all_entity_source_hubs,
        qualified_global_hub_uuids=queries.qualified_entity_global_hub_uuids,
        index_name='global_entity_hub_embedding',
        attach_global_hubs=writer.attach_entity_global_hubs,
        clear_invalid_global_hubs=writer.clear_invalid_entity_global_hubs,
        rebuild_global_names=lambda *a, **kw: _rebuild_global_names_callback(
            'entity', *a, **kw
        ),
    )
    records = await queries.all_entity_source_hubs(session_factory)
    result = await build_hubs(
        records,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
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


