import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kms2 import application
from kms2.config.settings import Settings
from kms2.core.model.source import Source
from kms2.core.model.teaching.session import (
    TeachingSessionSnapshot,
    TeachingStatus,
    TeachingTurnCommit,
    TeachingTurnRecord,
)
from kms2.core.model.user import Deck
from kms2.service.teaching import TeachingService


class _Runtime:
    def __init__(self, settings, *, embedding_token_counter=None):
        self.settings = settings
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


class _Reranker:
    async def rerank(self, query, documents, top_n=None):
        return []


@asynccontextmanager
async def _reranker_resources(settings, local_models, token_counter):
    yield _RERANKER


_RERANKER = _Reranker()


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
    settings = Settings(llm={'default_text_profile': 'qwen3.8-9b-distill-text'})
    settings.source_semantic.source_proposition_extraction.profile = (
        'chatgpt-luna'
    )
    settings.source_pedagogical.statement_procedure.role_typer.profile = (
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
        application._preflight_chatgpt(settings, settings.source_semantic)
    )
    assert calls == ['preflight']

    calls.clear()
    asyncio.run(
        application._preflight_chatgpt(settings, settings.source_learning)
    )
    assert calls == []


def test_chatgpt_preflight_skips_local_stage(monkeypatch):
    settings = Settings(llm={'default_text_profile': 'qwen3.8-9b-distill-text'})
    calls = []

    async def ensure_authenticated():
        calls.append('preflight')

    monkeypatch.setattr(
        application,
        'ensure_chatgpt_authenticated',
        ensure_authenticated,
    )

    asyncio.run(
        application._preflight_chatgpt(settings, settings.source_semantic)
    )
    assert calls == []


def test_chatgpt_auth_failure_stops_before_semantic_graph_build(
    monkeypatch,
):
    settings = Settings()
    settings.source_semantic.source_proposition_extraction.profile = (
        'chatgpt-luna'
    )

    async def fail_authentication():
        raise RuntimeError('ChatGPT auth unavailable')

    monkeypatch.setattr(
        application,
        'ensure_chatgpt_authenticated',
        fail_authentication,
    )
    monkeypatch.setattr(
        application,
        'build_source_semantic_graph',
        lambda *args, **kwargs: pytest.fail('graph must not start'),
    )

    with pytest.raises(RuntimeError, match='auth unavailable'):
        asyncio.run(
            application._run_source_semantic_stage(
                settings,
                None,
                'source-1',
            )
        )


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


class _SourceProcessingGraph:
    def __init__(self, fail=False):
        self.initial_state = None
        self.fail = fail

    async def ainvoke(self, initial_state):
        self.initial_state = initial_state
        if self.fail:
            raise RuntimeError('graph failed')
        return {
            'source': Source(uuid='result-source', key='book.pdf'),
            'ocr_pages': ['ocr-1'],
            'corrected_pages': ['corrected-1', 'corrected-2'],
            'formatted_pages': ['formatted-1', 'formatted-2', 'formatted-3'],
            'text_seam_pages': ['text-1'],
            'image_seam_pages': ['image-1', 'image-2'],
            'image_described_pages': ['described-1'],
            'embedded_pages': ['embedded-1', 'embedded-2'],
        }


class _SourcePedagogicalGraph:
    def __init__(self):
        self.initial_state = None

    async def ainvoke(self, initial_state):
        self.initial_state = initial_state
        return {
            'split_pages': ['page'],
            'instructions': ['instruction', 'instruction'],
            'exercise_components': ['exercise', 'exercise', 'exercise'],
            'pedagogical_components': [
                'component',
                'component',
                'component',
                'component',
            ],
            'statements': ['statement'] * 5,
            'procedures': ['procedure'] * 6,
            'reembedded_block_count': 7,
        }


class _ComposedGraph:
    def __init__(self, graph):
        self.graph = graph
        self.build_calls = 0

    def build_graph(self):
        self.build_calls += 1
        return self.graph


class _UserService:
    def __init__(self, database):
        self.database = database

    async def attach_new_source(self, user_uuid, source_uuid):
        return None

    async def list_sources(self, user_uuid):
        return [Source(uuid='result-source', key='book.pdf')]

    async def list_users(self):
        return []

    async def create_user(self, name):
        return SimpleNamespace(name=name)

    async def list_unowned_sources(self):
        return []

    async def adopt_source(self, user_uuid, source_uuid):
        return None


@pytest.fixture(autouse=True)
def _use_fake_user_service(monkeypatch):
    monkeypatch.setattr(application, 'build_user_service', _UserService)
    monkeypatch.setattr(application, 'reranker_resources', _reranker_resources)
    monkeypatch.setattr(
        application.InferenceTokenizers,
        'reranker',
        property(lambda tokenizers: object()),
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

    class _UserServiceSpy:
        def __init__(self, database):
            service_calls.append(('init', database))

        async def create_user(self, name):
            service_calls.append(('create', name))
            return user

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application.schema,
        'ensure_structural_schema',
        ensure_structural_schema,
    )
    monkeypatch.setattr(application, 'build_user_service', _UserServiceSpy)

    assert asyncio.run(application.create_user(settings, 'Alex')) is user
    assert schema_calls == [database.session]
    assert service_calls == [('init', database), ('create', 'Alex')]
    assert database.closed is True


def test_source_semantic_rejects_source_outside_user_before_runtime(
    monkeypatch,
):
    settings = Settings()
    database = _Database(settings.database)
    runtime_calls = []

    class _NoSources:
        def __init__(self, database):
            pass

        async def list_sources(self, user_uuid):
            assert user_uuid == 'user-1'
            return []

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(application, 'build_user_service', _NoSources)
    monkeypatch.setattr(
        application,
        'LocalModelRuntime',
        lambda _, **kwargs: runtime_calls.append(True),
    )

    with pytest.raises(ValueError, match='not owned by user'):
        asyncio.run(
            application.run_source_semantic_stage(
                settings,
                'user-1',
                'source-foreign',
            )
        )

    assert database.closed is True
    assert runtime_calls == []


def test_source_pedagogical_rejects_source_outside_user_before_runtime(
    monkeypatch,
):
    settings = Settings()
    database = _Database(settings.database)
    runtime_calls = []

    class _NoSources:
        def __init__(self, database):
            pass

        async def list_sources(self, user_uuid):
            assert user_uuid == 'user-1'
            return []

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(application, 'build_user_service', _NoSources)
    monkeypatch.setattr(
        application,
        'LocalModelRuntime',
        lambda _, **kwargs: runtime_calls.append(True),
    )

    with pytest.raises(ValueError, match='not owned by user'):
        asyncio.run(
            application.run_source_pedagogical_stage(
                settings,
                'user-1',
                'source-foreign',
            )
        )

    assert database.closed is True
    assert runtime_calls == []


def test_source_pedagogical_stage_runs_as_owned_standalone_stage(monkeypatch):
    settings = Settings(_env_file=None)

    async def ensure_authenticated():
        return None

    monkeypatch.setattr(
        application, 'ensure_chatgpt_authenticated', ensure_authenticated
    )
    source = Source(uuid='source-1', key='book.pdf')
    resources = SimpleNamespace(
        local_models=object(),
        embedding_client=object(),
        tokenizers=object(),
        database=object(),
        recorder=None,
    )
    graph = _SourcePedagogicalGraph()

    async def list_owned_sources(_settings, user_uuid):
        assert user_uuid == 'user-1'
        return [source]

    @asynccontextmanager
    async def application_resources(_settings, *, reranking_required=False):
        assert reranking_required is False
        yield resources

    monkeypatch.setattr(application, 'list_sources', list_owned_sources)
    monkeypatch.setattr(
        application, '_application_resources', application_resources
    )
    monkeypatch.setattr(
        application,
        'build_source_pedagogical_graph',
        lambda *args, **kwargs: _ComposedGraph(graph),
    )

    result = asyncio.run(
        application.run_source_pedagogical_stage(
            settings,
            'user-1',
            'source-1',
        )
    )

    assert result == application.SourcePedagogicalStageResult(
        split_page_count=1,
        instruction_count=2,
        exercise_component_count=3,
        pedagogical_component_count=4,
        statement_count=5,
        procedure_count=6,
        reembedded_block_count=7,
    )
    assert graph.initial_state == {'source_uuid': 'source-1'}


def test_combined_ingestion_runs_pedagogy_between_processing_and_semantics(
    monkeypatch,
):
    settings = Settings(_env_file=None)
    source = Source(uuid='result-source', key='book.pdf')
    resources = SimpleNamespace(
        local_models=object(),
        embedding_client=object(),
        tokenizers=object(),
        database=object(),
        recorder=None,
        reranker_client=object(),
    )
    events = []

    class _StageGraph:
        def __init__(self, stage, result):
            self._stage = stage
            self._result = result

        def build_graph(self):
            return self

        async def ainvoke(self, _initial_state):
            events.append(self._stage)
            return self._result

    processing_graph = _StageGraph(
        'processing',
        {
            'source': source,
            'ocr_pages': [],
            'corrected_pages': [],
            'formatted_pages': [],
            'text_seam_pages': [],
            'image_seam_pages': [],
            'image_described_pages': [],
            'embedded_pages': [],
        },
    )
    pedagogical_graph = _StageGraph(
        'pedagogical',
        {
            'split_pages': [],
            'instructions': [],
            'exercise_components': [],
            'pedagogical_components': [],
            'statements': [],
            'procedures': [],
            'reembedded_block_count': 0,
        },
    )
    semantic_graph = _StageGraph(
        'semantic',
        {
            'source_propositions': [],
            'source_proposition_hub_count': 0,
            'source_statement_hub_count': 0,
            'source_procedure_hub_count': 0,
            'source_entities': [],
            'source_entity_description_persisted_count': 0,
            'source_entity_hub_count': 0,
        },
    )

    async def ensure_authenticated():
        return None

    @asynccontextmanager
    async def application_resources(_settings, *, reranking_required=False):
        assert reranking_required is True
        events.append('resources-open')
        try:
            yield resources
        finally:
            events.append('resources-close')

    class _OwnershipService:
        def __init__(self, _database):
            pass

        async def attach_new_source(self, user_uuid, source_uuid):
            events.append(('attach', user_uuid, source_uuid))

    monkeypatch.setattr(
        application, 'ensure_chatgpt_authenticated', ensure_authenticated
    )
    monkeypatch.setattr(
        application, '_application_resources', application_resources
    )
    monkeypatch.setattr(
        application,
        'build_source_processing_graph',
        lambda *args, **kwargs: processing_graph,
    )
    monkeypatch.setattr(
        application,
        'build_source_pedagogical_graph',
        lambda *args, **kwargs: pedagogical_graph,
    )
    monkeypatch.setattr(
        application,
        'build_source_semantic_graph',
        lambda *args, **kwargs: semantic_graph,
    )
    monkeypatch.setattr(application, 'build_user_service', _OwnershipService)

    result = asyncio.run(
        application.ingest_and_run_source_semantic_stage(
            settings,
            'user-1',
            'book.pdf',
        )
    )

    assert events == [
        'resources-open',
        'processing',
        ('attach', 'user-1', 'result-source'),
        'pedagogical',
        'semantic',
        'resources-close',
    ]
    assert result.source_pedagogical_stage.statement_count == 0


def test_ingest_does_not_report_success_when_ownership_attach_fails(
    monkeypatch,
):
    settings = Settings(embedding={'backend': 'local'})
    monkeypatch.setattr(
        application, 'build_embedding_client', lambda *args: object()
    )
    database = _Database(settings.database)
    graph = _SourceProcessingGraph()
    events = []

    async def ensure_schema(session_factory, *, embedding_dimension):
        return None

    class _FailingOwnership:
        def __init__(self, database):
            pass

        async def attach_new_source(self, user_uuid, source_uuid):
            events.append(('attach', user_uuid, source_uuid))
            raise RuntimeError('ownership failed')

    monkeypatch.setattr(application.schema, 'ensure_schema', ensure_schema)
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(application, 'LocalModelRuntime', _Runtime)
    monkeypatch.setattr(application, 'build_user_service', _FailingOwnership)
    monkeypatch.setattr(
        application,
        'build_source_processing_graph',
        lambda *args, **kwargs: _ComposedGraph(graph),
    )

    with pytest.raises(RuntimeError, match='ownership failed'):
        asyncio.run(
            application.ingest_source(
                settings,
                'user-1',
                'fixtures/book.pdf',
            )
        )

    assert graph.initial_state is not None
    assert events == [('attach', 'user-1', 'result-source')]
    assert database.closed is True


def test_ingest_source_closes_database_when_graph_fails(monkeypatch):
    monkeypatch.setattr(
        application, 'build_embedding_client', lambda *args: object()
    )
    settings = Settings(embedding={'backend': 'local'})
    database = _Database(settings.database)
    graph = _SourceProcessingGraph(fail=True)

    monkeypatch.setattr(application, 'LocalModelRuntime', _Runtime)
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application,
        'build_source_processing_graph',
        lambda *args, **kwargs: _ComposedGraph(graph),
    )

    with pytest.raises(RuntimeError, match='graph failed'):
        asyncio.run(
            application.ingest_source(settings, 'user-1', 'fixtures/book.pdf')
        )

    assert database.closed is True


def test_list_sources_closes_database_and_returns_user_sources(monkeypatch):
    settings = Settings()
    database = _Database(settings.database)
    sources = [
        Source(uuid='source-1', key='book.pdf'),
        Source(uuid='source-2', key='notes.pdf'),
    ]
    repository_calls = []

    class _SourceService:
        def __init__(self, database):
            repository_calls.append(database)

        async def list_sources(self, user_uuid):
            assert user_uuid == 'user-1'
            return sources

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(application, 'build_user_service', _SourceService)

    result = asyncio.run(application.list_sources(settings, 'user-1'))

    assert result == sources
    assert len(repository_calls) == 1
    assert database.closed is True


def test_schema_failure_closes_database_before_starting_runtime(monkeypatch):
    settings = Settings(embedding={'backend': 'local'})
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    graph_built = []

    async def fail_schema(session_factory, *, embedding_dimension):
        raise RuntimeError('schema failed')

    monkeypatch.setattr(application.schema, 'ensure_schema', fail_schema)
    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application,
        'build_source_processing_graph',
        lambda *args, **kwargs: graph_built.append(True),
    )

    with pytest.raises(RuntimeError, match='schema failed'):
        asyncio.run(
            application.ingest_source(settings, 'user-1', 'fixtures/book.pdf')
        )

    assert database.closed is True
    assert runtime.entered is False
    assert graph_built == []


def test_semantic_cancellation_closes_reranker_before_runtime_and_database(
    monkeypatch,
):
    monkeypatch.setattr(
        application, 'build_embedding_client', lambda *args: object()
    )
    settings = Settings(embedding={'backend': 'local'})
    events = []

    @asynccontextmanager
    async def reranker_resources(settings, local_models, token_counter):
        try:
            yield _RERANKER
        finally:
            events.append('reranker-close')

    monkeypatch.setattr(application, 'reranker_resources', reranker_resources)

    class Runtime(_Runtime):
        async def __aexit__(self, exc_type, exc_value, traceback):
            events.append('runtime-close')

    class Database(_Database):
        async def close(self):
            events.append('database-close')

    runtime = Runtime(settings.local_models)
    database = Database(settings.database)
    started = asyncio.Event()

    class BlockingGraph:
        async def ainvoke(self, initial_state):
            started.set()
            await asyncio.Future()

    async def ensure_schema(session_factory, *, embedding_dimension):
        return None

    monkeypatch.setattr(application.schema, 'ensure_schema', ensure_schema)
    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    async def list_sources(settings, user_uuid):
        return [Source(uuid='result-source', key='book.pdf')]

    monkeypatch.setattr(application, 'list_sources', list_sources)
    monkeypatch.setattr(
        application,
        'build_source_semantic_graph',
        lambda *args, **kwargs: _ComposedGraph(BlockingGraph()),
    )

    async def exercise():
        task = asyncio.create_task(
            application.run_source_semantic_stage(
                settings, 'user-1', 'result-source'
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert events == ['reranker-close', 'runtime-close', 'database-close']


def test_list_decks_uses_database_without_local_runtime(monkeypatch):
    settings = Settings()
    database = _Database(settings.database)
    schema_calls = []
    service_calls = []

    async def ensure_structural_schema(session_factory):
        schema_calls.append(session_factory)

    class _LearningService:
        def __init__(self, database):
            service_calls.append(database)

        async def list_decks(self, user_uuid):
            assert user_uuid == 'user-1'
            return [Deck(uuid='deck-1', name='Default')]

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application.schema,
        'ensure_structural_schema',
        ensure_structural_schema,
    )
    monkeypatch.setattr(application, 'build_learning_service', _LearningService)

    decks = asyncio.run(application.list_decks(settings, 'user-1'))

    assert decks == [Deck(uuid='deck-1', name='Default')]
    assert schema_calls == [database.session]
    assert service_calls == [database]
    assert database.closed is True


class _ApplicationTeachingRepository:
    def __init__(self):
        self.snapshot = None

    async def create_session(self, _user_uuid, session):
        self.snapshot = TeachingSessionSnapshot(
            session=session,
            turns=[],
            current_assessment=None,
            current_plan=None,
            current_plan_acceptance=None,
            current_step_uuid=None,
            pending_question_uuid=None,
        )
        return deepcopy(self.snapshot)

    async def list_sessions(self, _user_uuid):
        return [deepcopy(self.snapshot.session)]

    async def load_session(self, _user_uuid, _session_uuid):
        return deepcopy(self.snapshot)

    async def append_response(
        self,
        _user_uuid,
        _session_uuid,
        expected_revision,
        turn,
    ):
        assert self.snapshot.session.revision == expected_revision
        self.snapshot.turns.append(
            TeachingTurnRecord(
                turn=turn,
                question=None,
                question_step_uuid=None,
                responds_to_question_uuid=self.snapshot.pending_question_uuid,
                presented_plan_uuid=None,
                accepted_plan_uuid=None,
            )
        )
        self.snapshot.session.status = TeachingStatus.AWAITING_TEACHER
        self.snapshot.session.revision += 1
        return deepcopy(self.snapshot)

    async def record_turn(
        self,
        _user_uuid,
        _session_uuid,
        expected_revision,
        turn,
        commit,
    ):
        assert self.snapshot.session.revision == expected_revision
        self.snapshot.turns.append(
            TeachingTurnRecord(
                turn=turn,
                question=commit.new_question,
                question_step_uuid=commit.new_question_step_uuid,
                responds_to_question_uuid=None,
                presented_plan_uuid=(
                    commit.new_plan.uuid
                    if commit.new_plan is not None
                    else None
                ),
                accepted_plan_uuid=None,
            )
        )
        self.snapshot.session.status = TeachingStatus.AWAITING_LEARNER
        self.snapshot.session.phase = commit.phase
        self.snapshot.session.revision += 1
        return deepcopy(self.snapshot)

    async def end_session(
        self,
        _user_uuid,
        _session_uuid,
        expected_revision,
        ended_at,
    ):
        assert self.snapshot.session.revision == expected_revision
        self.snapshot.pending_question_uuid = None
        self.snapshot.session.status = TeachingStatus.ENDED
        self.snapshot.session.updated_at = ended_at
        self.snapshot.session.ended_at = ended_at
        self.snapshot.session.revision += 1
        return deepcopy(self.snapshot)


def test_teaching_application_uses_database_only_resources_and_closes_them(
    monkeypatch,
):
    settings = Settings(_env_file=None)
    database = _Database(settings.database)
    schema_calls = []
    repository = _ApplicationTeachingRepository()
    service = TeachingService(repository)

    async def ensure_structural_schema(session_factory):
        schema_calls.append(session_factory)

    def tripwire(*args, **kwargs):
        raise AssertionError('inference resource must not be acquired')

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application.schema,
        'ensure_structural_schema',
        ensure_structural_schema,
    )
    monkeypatch.setattr(
        application, 'build_teaching_service', lambda _: service
    )
    monkeypatch.setattr(application, '_application_resources', tripwire)
    monkeypatch.setattr(application, 'LocalModelRuntime', tripwire)
    monkeypatch.setattr(application, 'InferenceTokenizers', tripwire)
    monkeypatch.setattr(application, 'build_embedding_client', tripwire)
    monkeypatch.setattr(application, 'reranker_resources', tripwire)

    started = asyncio.run(
        application.start_teaching_session(
            settings,
            'user-1',
            'Understand derivatives',
        )
    )
    loaded = asyncio.run(
        application.load_teaching_session(
            settings,
            'user-1',
            started.session.uuid,
        )
    )
    service_snapshot = asyncio.run(
        service.record_teaching_turn(
            'user-1',
            started.session.uuid,
            'Let us probe this goal.',
            TeachingTurnCommit(
                phase='probing',
                current_assessment_uuid=None,
                current_plan_uuid=None,
                current_step_uuid=None,
                pending_question_uuid=None,
            ),
            expected_revision=0,
        )
    )
    responded = asyncio.run(
        application.submit_teaching_response(
            settings,
            'user-1',
            started.session.uuid,
            'I am ready.',
            expected_revision=service_snapshot.session.revision,
        )
    )
    ended = asyncio.run(
        application.end_teaching_session(
            settings,
            'user-1',
            started.session.uuid,
            expected_revision=responded.session.revision,
        )
    )

    assert started.session.status is TeachingStatus.AWAITING_TEACHER
    assert loaded.session.revision == 0
    assert responded.session.status is TeachingStatus.AWAITING_TEACHER
    assert [turn.turn.content for turn in ended.turns] == [
        'Let us probe this goal.',
        'I am ready.',
    ]
    assert ended.session.status is TeachingStatus.ENDED
    assert ended.session.revision == 3
    assert len(schema_calls) == 4
    assert database.closed is True

    failure_database = _Database(settings.database)

    class _FailingRepository:
        async def load_session(self, _user_uuid, _session_uuid):
            raise RuntimeError('teaching repository failed')

    failing_service = TeachingService(_FailingRepository())

    monkeypatch.setattr(
        application,
        'DatabaseClient',
        lambda _: failure_database,
    )
    monkeypatch.setattr(
        application,
        'build_teaching_service',
        lambda _: failing_service,
    )
    with pytest.raises(RuntimeError, match='teaching repository failed'):
        asyncio.run(
            application.load_teaching_session(
                settings,
                'user-1',
                started.session.uuid,
            )
        )
    assert failure_database.closed is True
