import asyncio

from kms.construction import hub_inputs
from kms.core import models


def test_hub_input_node_builds_state_only_bundles():
    state = {
        'source_key': 'book-a',
        'entity_hub_components': [
            models.HubComponent(
                uuid='entity-a',
                source='book-a',
                node_id=1,
                name='graph',
                description='A graph.',
                embedding=[1.0, 0.0],
            )
        ],
        'predicate_hub_components': [
            models.HubComponent(
                uuid='predicate-a',
                source='book-a',
                node_id=1,
                name='contains',
                description='A containment relation.',
                embedding=[0.0, 1.0],
            )
        ],
        'entity_hub_records': [
            {
                'uuid': 'entity-hub-a',
                'canonical_name': 'graph',
                'aliases': ['$G'],
                'description': 'A mathematical graph.',
                'embedding': [1.0, 0.0],
                'source': 'book-a',
            }
        ],
        'predicate_hub_records': [],
    }

    result = asyncio.run(hub_inputs.HubInputNode().run(state))

    entity_bundle = result['entity_hub_bundle']
    predicate_bundle = result['predicate_hub_bundle']
    assert entity_bundle.source == 'book-a'
    assert entity_bundle.components[0].uuid == 'entity-a'
    assert entity_bundle.candidate_hubs[0].uuid == 'entity-hub-a'
    assert entity_bundle.candidate_hubs[0].aliases == ('$G',)
    assert predicate_bundle.source == 'book-a'
    assert predicate_bundle.components[0].uuid == 'predicate-a'
    assert predicate_bundle.candidate_hubs == ()


def test_hub_input_node_accepts_dict_components_without_graph():
    result = asyncio.run(
        hub_inputs.HubInputNode().run(
            {
                'source_key': 'book-a',
                'entity_hub_components': [
                    {
                        'uuid': 'entity-a',
                        'name': 'surface',
                        'description': 'Description',
                        'embedding': [1.0],
                        'source': 'book-a',
                    }
                ],
            }
        )
    )

    assert result['entity_hub_bundle'].components[0].name == 'surface'
    assert result['predicate_hub_bundle'].components == ()
