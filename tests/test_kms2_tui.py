import asyncio
import logging
import signal
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from kms2 import tui
from kms2.application import SourceSemanticStageResult
from kms2.config.settings import Settings
from kms2.core.model.learning import DueDeckCard, ReviewRating
from kms2.core.model.source import Source
from kms2.core.model.user import Deck, User


class _Prompt:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


def _semantic_result() -> SourceSemanticStageResult:
    return SourceSemanticStageResult(
        source_proposition_count=5,
        source_proposition_hub_count=2,
        source_statement_hub_count=0,
        source_procedure_hub_count=0,
        source_entity_count=0,
        source_entity_description_count=0,
        source_entity_hub_count=0,
    )


def _install_single_user(monkeypatch):
    user = User(uuid='user-1', name='Alex')

    async def list_users(settings):
        return [user]

    async def list_unowned_sources(settings):
        return []

    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    return user


def test_cli_login_runs_without_loading_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(tui.sys, 'argv', ['kms2', '--login-chatgpt'])
    monkeypatch.setattr(tui, 'login_chatgpt', lambda: calls.append('login'))
    monkeypatch.setattr(
        tui,
        'Settings',
        lambda: pytest.fail('login must not load application settings'),
    )

    tui.run()

    assert calls == ['login']


def test_cli_profile_selects_current_invocation_default(monkeypatch):
    settings = Settings()
    selected = []
    monkeypatch.setattr(
        tui.sys,
        'argv',
        ['kms2', '--llm-profile', 'chatgpt-luna'],
    )
    monkeypatch.setattr(tui, 'Settings', lambda: settings)
    monkeypatch.setattr(
        tui, '_run_tui', lambda current: selected.append(current)
    )

    tui.run()

    assert selected == [settings]
    assert settings.llm.default_text_profile == 'chatgpt-luna'


def test_tui_profile_selection_precedes_selected_inference(monkeypatch):
    settings = Settings()
    _install_single_user(monkeypatch)
    prompts = iter(
        [
            _Prompt(tui.SELECT_LLM_PROFILE_OPTION),
            _Prompt('chatgpt-luna'),
            _Prompt(tui.NEW_SOURCE_OPTION),
            _Prompt('book.pdf'),
            _Prompt(''),
        ]
    )
    ingestion_profiles = []

    async def list_sources(current_settings, user_uuid):
        return []

    async def ingest(current_settings, user_uuid, pdf_path, pages):
        ingestion_profiles.append(current_settings.llm.default_text_profile)
        return SimpleNamespace(
            source=Source(uuid='source-1', key=pdf_path),
            embedded_page_count=0,
        )

    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'ingest_source', ingest)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: next(prompts),
            filepath=lambda **kwargs: next(prompts),
            text=lambda **kwargs: next(prompts),
        ),
    )

    tui._run_tui(settings)

    assert ingestion_profiles == ['chatgpt-luna']


def test_tui_creates_first_user_before_new_source(monkeypatch):
    prompts = iter(
        [
            _Prompt('Alex'),
            _Prompt(tui.NEW_SOURCE_OPTION),
            _Prompt('book.pdf'),
            _Prompt(''),
        ]
    )
    source = Source(uuid='source-1', key='book.pdf')
    users = []
    actions = []
    ingested = []

    async def list_users(settings):
        return users

    async def create_user(settings, name):
        user = User(uuid='user-1', name=name)
        users.append(user)
        return user

    async def list_sources(settings, user_uuid):
        assert user_uuid == 'user-1'
        return []

    async def list_unowned_sources(settings):
        return []

    async def ingest(settings, user_uuid, pdf_path, pages):
        ingested.append((user_uuid, pdf_path, pages))
        return SimpleNamespace(
            source=source,
            embedded_page_count=1,
        )

    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            text=lambda **kwargs: next(prompts),
            select=lambda **kwargs: _record_select(actions, prompts, kwargs),
            filepath=lambda **kwargs: next(prompts),
        ),
    )
    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(tui, 'create_user', create_user)
    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    monkeypatch.setattr(tui, 'ingest_source', ingest)

    tui._run_tui()

    assert ingested == [('user-1', 'book.pdf', None)]
    assert actions[0]['choices'] == [
        tui.NEW_SOURCE_OPTION,
        tui.CREATE_DECK_OPTION,
        tui.ADD_CARD_OPTION,
        tui.REMOVE_CARD_OPTION,
        tui.REVIEW_CARDS_OPTION,
        tui.REVIEW_HISTORY_OPTION,
        tui.SELECT_LLM_PROFILE_OPTION,
    ]


def _record_select(calls, prompts, kwargs):
    calls.append(kwargs)
    return next(prompts)


def test_one_user_is_selected_without_prompt(monkeypatch):
    user = _install_single_user(monkeypatch)
    selections = []
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(select=lambda **kwargs: selections.append(kwargs)),
    )

    assert tui._select_user(Settings()) == user
    assert selections == []


def test_multiple_users_can_select_or_create(monkeypatch):
    users = [User(uuid='u1', name='Alex'), User(uuid='u2', name='Sam')]
    calls = []
    created = User(uuid='u3', name='Jo')

    async def list_users(settings):
        return users

    async def create_user(settings, name):
        assert name == 'Jo'
        return created

    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(tui, 'create_user', create_user)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _record_select(
                calls,
                iter([_Prompt(tui.CREATE_USER_OPTION)]),
                kwargs,
            ),
            text=lambda **kwargs: _Prompt('Jo'),
        ),
    )

    assert tui._select_user(Settings()) == created
    choices = calls[0]['choices']
    assert [choice.name for choice in choices[:2]] == [
        'Alex (u1)',
        'Sam (u2)',
    ]
    assert choices[2] == tui.CREATE_USER_OPTION


def test_multiple_users_can_select_existing_user(monkeypatch):
    users = [User(uuid='u1', name='Alex'), User(uuid='u2', name='Sam')]
    prompts = iter([_Prompt(users[1])])
    calls = []

    async def list_users(settings):
        return users

    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _record_select(calls, prompts, kwargs),
        ),
    )

    assert tui._select_user(Settings()) == users[1]
    assert calls[0]['message'] == 'Select a user:'


def test_existing_source_semantics_receive_selected_user(monkeypatch):
    user = _install_single_user(monkeypatch)
    source = Source(uuid='source-1', key='book.pdf')
    prompts = iter(
        [
            _Prompt(tui.EXISTING_SOURCE_OPTION),
            _Prompt(source),
            _Prompt(True),
        ]
    )
    calls = []
    choices = []

    async def list_sources(settings, user_uuid):
        assert user_uuid == user.uuid
        return [source]

    async def run_semantic(settings, user_uuid, source_uuid):
        calls.append((user_uuid, source_uuid))
        return _semantic_result()

    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'run_source_semantic_stage', run_semantic)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _record_select(choices, prompts, kwargs),
            confirm=lambda **kwargs: next(prompts),
        ),
    )

    tui._run_tui()

    assert choices[0]['choices'] == [
        tui.NEW_SOURCE_OPTION,
        tui.CREATE_DECK_OPTION,
        tui.ADD_CARD_OPTION,
        tui.REMOVE_CARD_OPTION,
        tui.REVIEW_CARDS_OPTION,
        tui.REVIEW_HISTORY_OPTION,
        tui.SELECT_LLM_PROFILE_OPTION,
        tui.SOURCE_PEDAGOGICAL_OPTION,
        tui.EXISTING_SOURCE_OPTION,
        tui.SOURCE_LEARNING_OPTION,
    ]
    assert calls == [(user.uuid, source.uuid)]


def test_existing_source_pedagogical_stage_receives_selected_user(monkeypatch):
    user = _install_single_user(monkeypatch)
    source = Source(uuid='source-1', key='book.pdf')
    prompts = iter(
        [
            _Prompt(tui.SOURCE_PEDAGOGICAL_OPTION),
            _Prompt(source),
            _Prompt(True),
        ]
    )
    choices = []
    calls = []

    async def list_sources(settings, user_uuid):
        assert user_uuid == user.uuid
        return [source]

    async def run_pedagogical(settings, user_uuid, source_uuid):
        calls.append((user_uuid, source_uuid))
        return SimpleNamespace(
            split_page_count=1,
            instruction_count=1,
            exercise_component_count=1,
            pedagogical_component_count=0,
            statement_count=1,
            procedure_count=0,
            reembedded_block_count=0,
        )

    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'run_source_pedagogical_stage', run_pedagogical)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _record_select(choices, prompts, kwargs),
            confirm=lambda **kwargs: next(prompts),
        ),
    )

    tui._run_tui()

    assert calls == [(user.uuid, source.uuid)]
    assert choices[0]['choices'][7:10] == [
        tui.SOURCE_PEDAGOGICAL_OPTION,
        tui.EXISTING_SOURCE_OPTION,
        tui.SOURCE_LEARNING_OPTION,
    ]


def test_tui_offers_explicit_adoption_only_for_unowned_sources(monkeypatch):
    user = _install_single_user(monkeypatch)
    source = Source(uuid='legacy-1', key='legacy.pdf')
    actions = []
    prompts = iter([_Prompt(tui.ADOPT_SOURCE_OPTION), _Prompt(source)])
    adopted = []

    async def list_sources(settings, user_uuid):
        return []

    async def list_unowned_sources(settings):
        return [source]

    async def adopt(settings, user_uuid, source_uuid):
        adopted.append((user_uuid, source_uuid))

    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    monkeypatch.setattr(tui, 'adopt_source', adopt)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _record_select(actions, prompts, kwargs),
            confirm=lambda **kwargs: _Prompt(True),
        ),
    )

    tui._run_tui()

    assert actions[0]['choices'] == [
        tui.NEW_SOURCE_OPTION,
        tui.CREATE_DECK_OPTION,
        tui.ADD_CARD_OPTION,
        tui.REMOVE_CARD_OPTION,
        tui.REVIEW_CARDS_OPTION,
        tui.REVIEW_HISTORY_OPTION,
        tui.SELECT_LLM_PROFILE_OPTION,
        tui.ADOPT_SOURCE_OPTION,
    ]
    assert adopted == [(user.uuid, source.uuid)]


def test_pdf_directory_is_created_under_working_directory(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    directory = tui._pdf_directory()

    assert directory == tmp_path / 'pdfs'
    assert directory.is_dir()


def test_tui_reports_cancelled_pipeline(monkeypatch, caplog):
    def cancel(settings=None):
        raise asyncio.CancelledError

    monkeypatch.setattr(tui, '_run_tui', cancel)
    caplog.set_level(logging.INFO, logger='kms2.tui')
    monkeypatch.setattr(tui.sys, 'argv', ['kms2'])
    with pytest.raises(SystemExit) as exit_info:
        tui.run()

    assert exit_info.value.code == 0
    assert 'Cancelled.' in caplog.messages


def test_run_async_cancels_pipeline_on_sigint():
    async def pipeline():
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(60)

    with pytest.raises(asyncio.CancelledError):
        tui._run_async(pipeline())


def test_review_due_cards_sends_selected_rating_and_utc_time(monkeypatch):
    settings = Settings()
    user = User(uuid='user-1', name='Alex')
    deck = Deck(uuid='deck-1', name='Default')
    card = DueDeckCard(
        card_uuid='card-1',
        question='Question',
        answer='Answer',
        due_at=datetime.now(UTC),
    )
    review_calls = []

    async def due_cards(settings, user_uuid, deck_uuid, now):
        assert user_uuid == user.uuid
        assert deck_uuid == deck.uuid
        assert now.tzinfo is not None
        return [card]

    async def record_review(
        settings,
        user_uuid,
        deck_uuid,
        card_uuid,
        rating,
        review_datetime,
        review_duration=None,
    ):
        review_calls.append(
            (
                user_uuid,
                deck_uuid,
                card_uuid,
                rating,
                review_datetime,
                review_duration,
            )
        )

    monkeypatch.setattr(tui, '_select_deck', lambda *args: deck)
    monkeypatch.setattr(tui, 'list_due_deck_cards', due_cards)
    monkeypatch.setattr(tui, 'review_card', record_review)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _Prompt(ReviewRating.GOOD),
        ),
    )

    tui._review_due_cards(settings, user)

    assert review_calls[0][0:4] == (
        user.uuid,
        deck.uuid,
        card.card_uuid,
        ReviewRating.GOOD,
    )
    assert review_calls[0][4].utcoffset().total_seconds() == 0
