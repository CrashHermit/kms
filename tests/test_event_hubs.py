import asyncio
from importlib import import_module

from kms.construction import local_event_hubs, triplet_extractor
from kms.core import models
from kms.graph import hubs

global_event_hubs = import_module('kms.postprocessing.global.event_hubs')


def test_triplet_prompt_separates_events_from_results():
    prompt = triplet_extractor._TripletSignature.__doc__
    assert 'full clause, explanatory phrase' in prompt
    assert 'shortest complete noun phrase' in prompt
    assert 'Never infer events' in prompt


def test_event_adjudication_prompt_rejects_related_events():
    prompt = global_event_hubs.EventHubAdjudicationSignature.__doc__
    assert 'Return FALSE when one is a consequence' in prompt
    assert 'battle and its victory are distinct events' in prompt
    assert 'Shared verbs or' in prompt


def test_event_synthesis_prompt_preserves_occurrence_boundaries():
    prompt = local_event_hubs.EventHubSynthesisSignature.__doc__
    assert 'not its consequence' in prompt
    assert 'Never invent time, participants' in prompt


def test_event_hub_node_normalizes_components_and_updates_bundle(monkeypatch):
    captured = {}

    async def build(records, *, source, spec, synthesizer):
        captured['records'] = records
        captured['source'] = source
        captured['spec'] = spec
        captured['synthesizer'] = synthesizer
        return {
            'assignments': [
                {'component': 'event-component', 'hub': 'event-hub'}
            ],
            'diagnostics': {'records': 1, 'final_hubs': 1},
            'hubs': [{'uuid': 'event-hub', 'members': ['event-component']}],
        }

    monkeypatch.setattr(local_event_hubs, 'build_event_hubs', build)
    synthesizer = object()
    component = models.HubComponent(
        uuid='event-component',
        source='source',
        node_id=3,
        name='appointment',
        description='An appointment occurrence.',
        embedding=[1.0, 0.0],
    )
    result = asyncio.run(
        local_event_hubs.EventHubNode(synthesizer).run(
            {
                'source': models.Source(key='source'),
                'event_hub_components': [component],
            }
        )
    )

    assert captured['source'] == 'source'
    assert captured['synthesizer'] is synthesizer
    assert captured['records'] == [
        {
            'uuid': 'event-component',
            'name': 'appointment',
            'description': 'An appointment occurrence.',
            'embedding': [1.0, 0.0],
            'source': 'source',
            'aliases': [],
        }
    ]
    assert captured['spec'].hub_id_factory(
        captured['records'], {}
    ) == hubs.hub_uuid(hubs.LOCAL_EVENT_HUB_LABEL, 'source', 'event-component')
    assert result['event_hub_assignments'] == [
        {'component': 'event-component', 'hub': 'event-hub'}
    ]
    assert result['event_hub_diagnostics'] == {'records': 1, 'final_hubs': 1}
    assert result['event_hub_records'] == [
        {'uuid': 'event-hub', 'members': ['event-component']}
    ]
    assert (
        result['construction_bundle'].event_hub_assignments
        == result['event_hub_assignments']
    )
