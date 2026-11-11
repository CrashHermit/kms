import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from kms import application
from kms.config.llm import LocalLLMProfile
from kms.config.settings import Settings
from kms.core.embedding import EmbeddingModel
from kms.source.models.source import Source


class _Runtime:
    def __init__(self, settings):
        self.settings = settings
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True


_EMBEDDINGS = SimpleNamespace(
    embedder=object(),
    model=EmbeddingModel(
        model='voyage/voyage-multimodal-3.5',
        dimension=1024,
    ),
    budget=object(),
)


@asynccontextmanager
async def _embedding_resources(settings):
    yield _EMBEDDINGS


def test_application_chatgpt_login_delegates_to_provider(monkeypatch):
    calls = []
    monkeypatch.setattr(
        application,
        '_login_chatgpt',
        lambda: calls.append('login'),
    )

    assert application.login_chatgpt() is None
    assert calls == ['login']


def test_chatgpt_preflight_resolves_only_relevant_stage_settings(
    monkeypatch,
):
    settings = Settings(
        _env_file=None, llm={'default_text_profile': 'qwen3.8-9b-distill-text'}
    )
    for stage in (
        settings.source_processing.content_correction,
        settings.source_processing.image_seam,
        settings.source_processing.image_filter,
    ):
        stage.inference.profile = 'gemma-vision-8k'
    settings.source_processing.formatting.inference.profile = (
        'qwen3.8-9b-distill-text'
    )
    settings.source_processing.statement_procedure.role_typer.profile = (
        'chatgpt-luna'
    )
    calls = []

    async def ensure_authenticated():
        calls.append('preflight')

    monkeypatch.setattr(
        application,
        'ensure_chatgpt_authenticated',
        ensure_authenticated,
    )

    asyncio.run(
        application._preflight_chatgpt(
            settings, settings.source_processing.statement_procedure
        )
    )
    assert calls == ['preflight']

    calls.clear()
    asyncio.run(
        application._preflight_chatgpt(
            settings, settings.source_processing.formatting
        )
    )
    assert calls == []


def test_chatgpt_preflight_skips_local_stage(monkeypatch):
    settings = Settings(
        _env_file=None, llm={'default_text_profile': 'qwen3.8-9b-distill-text'}
    )
    # The named vision profile and application text default both run locally.
    settings.llm.profiles['chatgpt-luna'] = LocalLLMProfile(
        model_server_profile='gemma-vision-8k'
    )
    calls = []

    async def ensure_authenticated():
        calls.append('preflight')

    monkeypatch.setattr(
        application,
        'ensure_chatgpt_authenticated',
        ensure_authenticated,
    )

    asyncio.run(
        application._preflight_chatgpt(settings, settings.source_processing)
    )
    assert calls == []


class _Database:
    def __init__(self, settings):
        self.settings = settings
        self.closed = False
        self.schema_calls = []

    async def close(self):
        self.closed = True

    def session(self):
        return _SchemaSession(self.schema_calls)


class _SchemaSession:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def run(self, statement, **parameters):
        self.calls.append((statement, parameters))
        return self

    async def data(self):
        return []

    async def consume(self):
        pass


@pytest.fixture(autouse=True)
def _use_fake_resources(monkeypatch):
    monkeypatch.setattr(
        application, 'embedding_resources', _embedding_resources
    )


def test_create_user_ensures_structural_schema_and_closes_database(
    monkeypatch,
):
    settings = Settings()
    database = _Database(settings.database)
    schema_calls = []
    service_calls = []
    user = SimpleNamespace(name='Alex')

    async def ensure_structural_schema(session_factory):
        schema_calls.append(session_factory)

    class _UserRepositorySpy:
        def __init__(self, session_factory):
            service_calls.append(('init', session_factory))

        async def create_user(self, user):
            service_calls.append(('create', user.name))

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application.schema,
        'ensure_structural_schema',
        ensure_structural_schema,
    )
    monkeypatch.setattr(application, 'UserRepository', _UserRepositorySpy)

    created = asyncio.run(application.create_user(settings, 'Alex'))
    assert created.name == user.name
    assert schema_calls == [database.session]
    assert service_calls == [('init', database.session), ('create', 'Alex')]
    assert database.closed is True


def test_unknown_user_fails_before_provider_acquisition(monkeypatch):
    @asynccontextmanager
    async def database_resources(settings):
        yield SimpleNamespace(session=object())

    class MissingUser:
        def __init__(self, session):
            pass

        async def read_user(self, user_uuid):
            raise LookupError(user_uuid)

    def unexpected_resources(*args, **kwargs):
        raise AssertionError('provider resources acquired for missing user')

    monkeypatch.setattr(application, '_database_resources', database_resources)
    monkeypatch.setattr(application, 'UserRepository', MissingUser)
    monkeypatch.setattr(
        application, '_application_resources', unexpected_resources
    )
    with pytest.raises(LookupError, match='missing-user'):
        settings = Settings(_env_file=None)
        capture = application.CapturedPDFSource(
            data=b'pdf', retrieved_at=application.datetime.now(application.UTC)
        )
        asyncio.run(
            application.ingest_captured_source(
                settings, 'missing-user', capture
            )
        )


def test_list_sources_closes_database_and_returns_user_sources(monkeypatch):
    settings = Settings()
    database = _Database(settings.database)
    sources = [
        Source(uuid='source-1', key='book.pdf'),
        Source(uuid='source-2', key='notes.pdf'),
    ]
    repository_calls = []

    class _SourceRepository:
        def __init__(self, session_factory):
            repository_calls.append(session_factory)

        async def list_sources(self, user_uuid):
            assert user_uuid == 'user-1'
            return sources

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application, 'SourceCatalogRepository', _SourceRepository
    )

    result = asyncio.run(application.list_sources(settings, 'user-1'))

    assert result == sources
    assert len(repository_calls) == 1
    assert database.closed is True


def test_schema_failure_closes_resources_and_database(monkeypatch):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)

    async def fail_schema(session_factory, *, embedding_dimension):
        assert embedding_dimension == _EMBEDDINGS.model.dimension
        raise RuntimeError('schema failed')

    monkeypatch.setattr(application.schema, 'ensure_schema', fail_schema)
    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda settings: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    async def acquire():
        async with application._application_resources(settings):
            pass

    with pytest.raises(RuntimeError, match='schema failed'):
        asyncio.run(acquire())

    assert database.closed is True
    assert runtime.entered is True
    assert runtime.closed is True


def test_search_sources_with_empty_allowlist_skips_application_resources(
    monkeypatch,
) -> None:
    settings = Settings(_env_file=None)

    def fail_resources(*_args, **_kwargs):
        pytest.fail('empty source scope must not acquire resources')

    monkeypatch.setattr(application, '_application_resources', fail_resources)

    assert (
        asyncio.run(
            application.search_sources(
                settings,
                'user-1',
                'query',
                source_uuids=[],
            )
        )
        == []
    )
