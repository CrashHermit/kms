import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kms import maintenance
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
