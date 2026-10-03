import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from kms2 import application, tui
from kms2.config.settings import Settings
from kms2.core.model.source import Source
from kms2.core.model.user import User


def test_source_learning_application_rejects_unowned_source(monkeypatch):
    async def list_sources(settings, user_uuid):
        return []

    monkeypatch.setattr(application, 'list_sources', list_sources)

    with pytest.raises(ValueError, match='not owned by user'):
        asyncio.run(
            application.run_source_learning_stage(
                Settings(), 'user-1', 'foreign-source'
            )
        )


def test_tui_exposes_owned_source_learning_action(monkeypatch):
    user = User(uuid='user-1', name='Alex')
    source = Source(uuid='source-1', key='book.pdf')
    prompts = iter([tui.SOURCE_LEARNING_OPTION, source, True])
    selections = []
    calls = []

    class Prompt:
        def execute(self):
            return next(prompts)

    async def list_users(settings):
        return [user]

    async def list_sources(settings, user_uuid):
        return [source]

    async def list_unowned_sources(settings):
        return []

    async def run_learning(settings, user_uuid, source_uuid):
        calls.append((user_uuid, source_uuid))
        return SimpleNamespace(
            entity_learning_fact_count=2,
            entity_flashcard_count=2,
            event_learning_fact_count=0,
            event_flashcard_count=0,
            predicate_learning_fact_count=0,
            predicate_flashcard_count=0,
            triplet_learning_fact_count=0,
            triplet_flashcard_count=0,
        )

    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    monkeypatch.setattr(tui, 'run_source_learning_stage', run_learning)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: selections.append(kwargs) or Prompt(),
            confirm=lambda **kwargs: Prompt(),
        ),
    )

    tui._run_tui()

    assert [choice for choice in selections[0]['choices']] == [
        tui.NEW_SOURCE_OPTION,
        tui.CREATE_DECK_OPTION,
        tui.ADD_CARD_OPTION,
        tui.REMOVE_CARD_OPTION,
        tui.REVIEW_CARDS_OPTION,
        tui.REVIEW_HISTORY_OPTION,
        tui.EXISTING_SOURCE_OPTION,
        tui.SOURCE_LEARNING_OPTION,
        tui.GLOBAL_SEMANTIC_OPTION,
    ]
    assert calls == [('user-1', 'source-1')]


def test_source_learning_application_runs_owned_source(monkeypatch):
    source = Source(uuid='source-1', key='book.pdf')
    resources = SimpleNamespace(
        local_models=object(),
        database=object(),
        recorder=None,
    )
    calls = []

    async def list_sources(settings, user_uuid):
        return [source]

    class Graph:
        async def ainvoke(self, initial_state):
            calls.append(initial_state)
            return {
                'entity_learning_fact_count': 2,
                'entity_flashcard_count': 2,
                'event_learning_fact_count': 1,
                'event_flashcard_count': 1,
                'predicate_learning_fact_count': 3,
                'predicate_flashcard_count': 3,
                'triplet_learning_fact_count': 4,
                'triplet_flashcard_count': 4,
            }

    @asynccontextmanager
    async def application_resources(settings):
        yield resources

    monkeypatch.setattr(application, 'list_sources', list_sources)
    monkeypatch.setattr(
        application, '_application_resources', application_resources
    )
    monkeypatch.setattr(
        application,
        'build_source_learning_graph',
        lambda *args, **kwargs: SimpleNamespace(build_graph=lambda: Graph()),
    )

    result = asyncio.run(
        application.run_source_learning_stage(Settings(), 'user-1', source.uuid)
    )

    assert result.entity_learning_fact_count == 2
    assert result.triplet_flashcard_count == 4
    assert calls == [{'source_uuid': source.uuid}]
