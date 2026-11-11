import asyncio
import logging
import signal
from types import SimpleNamespace

import pytest

from kms import tui
from kms.config.settings import Settings
from kms.source.models.source import Source
from kms.users.models import User


class _Prompt:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


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
    monkeypatch.setattr(tui.sys, 'argv', ['kms', '--login-chatgpt'])
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
        ['kms', '--llm-profile', 'chatgpt-luna'],
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

    async def ingest(current_settings, user_uuid, pdf_path, pages):
        ingestion_profiles.append(current_settings.llm.default_text_profile)
        return SimpleNamespace(
            source=Source(uuid='source-1', key=pdf_path),
        )

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

    async def list_unowned_sources(settings):
        return []

    async def ingest(settings, user_uuid, pdf_path, pages):
        ingested.append((user_uuid, pdf_path, pages))
        return SimpleNamespace(
            source=source,
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
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    monkeypatch.setattr(tui, 'ingest_source', ingest)

    tui._run_tui()

    assert ingested == [('user-1', 'book.pdf', None)]


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


def test_tui_offers_explicit_adoption_only_for_unowned_sources(monkeypatch):
    user = _install_single_user(monkeypatch)
    source = Source(uuid='legacy-1', key='legacy.pdf')
    actions = []
    prompts = iter([_Prompt(tui.ADOPT_SOURCE_OPTION), _Prompt(source)])
    adopted = []

    async def list_unowned_sources(settings):
        return [source]

    async def adopt(settings, user_uuid, source_uuid):
        adopted.append((user_uuid, source_uuid))

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

    assert adopted == [(user.uuid, source.uuid)]


@pytest.mark.parametrize('has_results', [True, False])
def test_tui_discovery_displays_links_without_ingestion(
    monkeypatch, caplog, has_results
):
    _install_single_user(monkeypatch)
    candidate = SimpleNamespace(
        title='Useful', url='https://example.com', snippet='Summary'
    )
    questions = []

    async def discover(settings, question):
        questions.append(question)
        return [candidate] if has_results else []

    monkeypatch.setattr(tui, 'discover_sources', discover)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: _Prompt(tui.DISCOVERY_OPTION),
            text=lambda **kwargs: _Prompt('question'),
        ),
    )
    with caplog.at_level(logging.INFO):
        tui._run_tui(Settings(_env_file=None))
    assert questions == ['question']
    if has_results:
        assert 'Useful\nhttps://example.com\nSummary' in caplog.messages
    else:
        assert 'No sources selected.' in caplog.messages


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
    caplog.set_level(logging.INFO, logger='kms.tui')
    monkeypatch.setattr(tui.sys, 'argv', ['kms'])
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
