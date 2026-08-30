"""Build cross-source global EventHubs from local EventHubs."""

from kms.construction import local_event_hubs as local_hubs
from kms.graph import local_event_hubs as graph_hubs
from kms.graph import queries, writer

from . import entity_hubs

EventHubAdjudicator = local_hubs.EventHubAdjudicator
EventHubSynthesizer = local_hubs.EventHubSynthesizer


async def align_global_hubs(
    *,
    session_factory,
    language_model,
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Align local event hubs into deterministic cross-source EventHubs."""
    records = await queries.all_event_source_hubs(session_factory)
    sources = {record.get('source') for record in records if record.get('source')}
    if len(sources) < 2:
        return {'aligned': 0, 'new_hubs': 0}
    spec = entity_hubs.GlobalEntityHubSpec(
        graph=graph_hubs,
        stage_name='event_hubs',
        hub_id_factory=entity_hubs.global_hub_id_factory(graph_hubs),
        source_resolver=lambda _: None,
        adjudication_context='Compare local event hubs across different sources by meaning and roles.',
        synthesis_context='Synthesize a cross-source semantic event hub.',
        all_components=None,
        clear_hubs=writer.clear_event_global_hubs,
        persist_hubs=writer.persist_event_hubs,
        rebuild_names=None,
        rebuild_triplets=None,
        all_source_hubs=queries.all_event_source_hubs,
        qualified_global_hub_uuids=queries.qualified_event_global_hub_uuids,
        index_name='global_event_hub_embedding',
        attach_global_hubs=writer.attach_event_global_hubs,
        clear_invalid_global_hubs=writer.clear_invalid_event_global_hubs,
        rebuild_global_names=None,
    )
    component_records = entity_hubs.local_hub_records(records)
    result = await entity_hubs.build_hubs(
        component_records,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    result['hubs'] = [
        hub for hub in result['hubs']
        if len({member.get('source') for member in component_records if member['uuid'] in hub.get('members', [])}) >= 2
    ]
    await writer.clear_event_global_hubs(session_factory=session_factory)
    await writer.persist_event_hubs(
        result['hubs'], session_factory=session_factory, tier='meta'
    )
    assignments = [
        {'source_hub': member, 'meta_hub': hub['uuid']}
        for hub in result['hubs']
        for member in hub.get('members', [])
    ]
    await writer.attach_event_global_hubs(
        assignments, aliases=[], session_factory=session_factory
    )
    return {'aligned': len(assignments), 'new_hubs': len(result['hubs'])}
