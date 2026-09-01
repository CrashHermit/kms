import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kms import maintenance
from kms.construction import maintenance as construction_maintenance
from kms.core import models


@pytest.mark.parametrize(
    ('command', 'kind'),
    [
        ('generate-entity-cards', models.CardTargetKind.ENTITY),
        ('generate-event-cards', models.CardTargetKind.EVENT),
        ('generate-triplet-cards', models.CardTargetKind.TRIPLET),
        ('generate-procedure-cards', models.CardTargetKind.PROCEDURE),
    ],
)
def test_card_commands_route_to_matching_kind(monkeypatch, command, kind):
    calls = []
    designer = object()

    async def create_cards_for_hub(**kwargs):
        calls.append(kwargs)
        return {'target_kind': kwargs['target_kind'].value}

    monkeypatch.setattr(
        maintenance.cards, 'create_cards_for_hub', create_cards_for_hub
    )
    monkeypatch.setattr(
        maintenance, '_card_designer', lambda selected: designer
    )

    result = asyncio.run(
        maintenance._generate_cards(
            session_factory=object(),
            target_kind=kind,
            hub_uuid='hub-id',
            designer=designer,
        )
    )

    assert result == {'target_kind': kind.value}
    assert calls == [
        {
            'session_factory': calls[0]['session_factory'],
            'target_kind': kind,
            'hub_uuid': 'hub-id',
            'designer': designer,
        }
    ]


@pytest.mark.parametrize(
    'command',
    [
        'generate-entity-cards',
        'generate-event-cards',
        'generate-triplet-cards',
        'generate-procedure-cards',
    ],
)
def test_card_command_accepts_only_hub_uuid(command):
    arguments = maintenance._parser().parse_args([command, 'hub-id'])
    assert arguments.command == command
    assert arguments.hub_uuid == 'hub-id'


def test_generic_card_command_removed():
    with pytest.raises(SystemExit):
        maintenance._parser().parse_args(
            ['generate-hub-cards', 'entity', 'hub-id']
        )


def test_generate_entity_cards_does_not_build_aggregate_models(monkeypatch):
    class Application:
        model_manager = object()

        def session_factory(self):
            return object()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    calls = []
    monkeypatch.setattr(maintenance.runtime, 'Runtime', Application)
    monkeypatch.setattr(
        maintenance.serve,
        'model_manager_context',
        lambda manager: nullcontext(),
    )
    monkeypatch.setattr(
        maintenance,
        '_card_designer',
        lambda kind: calls.append(kind) or object(),
    )
    monkeypatch.setattr(
        maintenance,
        '_models',
        lambda: pytest.fail('aggregate models were built'),
    )

    async def create_cards_for_hub(**kwargs):
        return {'target_kind': kwargs['target_kind'].value}

    monkeypatch.setattr(
        maintenance.cards, 'create_cards_for_hub', create_cards_for_hub
    )

    result = asyncio.run(
        maintenance._run(
            SimpleNamespace(
                command='generate-entity-cards',
                hub_uuid='hub-id',
            )
        )
    )

    assert result == {'target_kind': 'entity'}
    assert calls == [models.CardTargetKind.ENTITY]


def test_rebuild_source_orders_local_hub_stages(monkeypatch):
    calls = []

    def stage(name, result):
        async def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return run

    monkeypatch.setattr(
        construction_maintenance.local_entity_hubs,
        'rebuild',
        stage('entity', {'clusters': 1}),
    )
    monkeypatch.setattr(
        construction_maintenance.local_predicate_hubs,
        'rebuild',
        stage('predicate', {'clusters': 2}),
    )
    monkeypatch.setattr(
        construction_maintenance.local_event_hubs,
        'rebuild',
        stage('event', {'clusters': 3}),
    )
    monkeypatch.setattr(
        construction_maintenance.triplet_hubs,
        'rebuild',
        stage('triplet', {'triplet_hubs': 4}),
    )
    monkeypatch.setattr(
        construction_maintenance.local_statement_hubs,
        'rebuild',
        stage('statement', {'clusters': 5}),
    )
    monkeypatch.setattr(
        construction_maintenance.local_procedure_hubs,
        'rebuild',
        stage('procedure', {'clusters': 6}),
    )

    triplet_model = object()
    result = asyncio.run(
        construction_maintenance.rebuild_source(
            'book-a',
            session_factory=object(),
            entity_language_model=object(),
            entity_synthesizer=object(),
            predicate_language_model=object(),
            predicate_synthesizer=object(),
            event_synthesizer=object(),
            triplet_language_model=triplet_model,
            statement_synthesizer=object(),
            procedure_synthesizer=object(),
            max_concurrency=8,
        )
    )

    assert [name for name, _, _ in calls] == [
        'entity',
        'predicate',
        'event',
        'triplet',
        'statement',
        'procedure',
    ]
    triplet_call = calls[3]
    assert triplet_call[2]['language_model'] is triplet_model
    assert triplet_call[2]['source'] == 'book-a'
    assert calls[4][2]['max_concurrency'] == 8
    assert calls[5][2]['max_concurrency'] == 8
    assert triplet_call[2]['max_concurrency'] == 8
    assert list(result) == [
        'entity',
        'predicate',
        'event',
        'triplet',
        'statement',
        'procedure',
    ]


def test_rebuild_source_cli_forwards_event_and_triplet_models(monkeypatch):
    class Application:
        model_manager = object()

        def session_factory(self):
            return object()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    entity_model = object()
    predicate_model = object()
    event_model = object()
    triplet_model = object()
    captured = {}

    monkeypatch.setattr(maintenance.runtime, 'Runtime', Application)
    monkeypatch.setattr(
        maintenance.serve,
        'model_manager_context',
        lambda manager: nullcontext(),
    )
    monkeypatch.setattr(
        maintenance,
        '_models',
        lambda: {
            'entity_language_model': entity_model,
            'entity_synthesizer': 'entity-synth',
            'predicate_language_model': predicate_model,
            'predicate_synthesizer': 'predicate-synth',
            'event_synthesizer': 'event-synth',
            'event_language_model': event_model,
            'triplet_language_model': triplet_model,
            'statement_synthesizer': 'statement-synth',
            'procedure_synthesizer': 'procedure-synth',
        },
    )

    async def rebuild_source(*args, **kwargs):
        captured.update(kwargs)
        return {'triplet': {'triplet_hubs': 1}}

    monkeypatch.setattr(
        maintenance.maintenance, 'rebuild_source', rebuild_source
    )
    monkeypatch.setattr(
        maintenance.local_event_hubs,
        'rebuild',
        lambda *args, **kwargs: pytest.fail(
            'event hubs must be rebuilt by construction maintenance'
        ),
    )

    result = asyncio.run(
        maintenance._run(
            SimpleNamespace(command='rebuild-source', source='book-a')
        )
    )

    assert result == {'triplet': {'triplet_hubs': 1}}
    assert captured['event_synthesizer'] == 'event-synth'
    assert captured['triplet_language_model'] is triplet_model
    assert captured['statement_synthesizer'] == 'statement-synth'
    assert captured['procedure_synthesizer'] == 'procedure-synth'
    assert 'statement_adjudicator' not in captured
    assert 'procedure_adjudicator' not in captured
