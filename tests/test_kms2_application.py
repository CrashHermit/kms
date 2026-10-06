import asyncio
from types import SimpleNamespace

import pytest

from kms2 import application
from kms2.config.services import TrainingSettings
from kms2.config.settings import Settings
from kms2.core.model.source import Source
from kms2.core.model.user import Deck


class _Runtime:
    def __init__(self, settings, *, embedding_token_counter=None):
        self.settings = settings
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


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
            'split_pages': ['split-1', 'split-2', 'split-3'],
            'instructions': ['instruction-1', 'instruction-2'],
            'exercise_components': ['exercise-1'],
            'pedagogical_components': ['pedagogical-1', 'pedagogical-2'],
            'statements': ['statement-1'],
            'procedures': ['procedure-1', 'procedure-2', 'procedure-3'],
            'embedded_pages': ['embedded-1', 'embedded-2'],
        }


class _SourceSemanticGraph:
    async def ainvoke(self, initial_state):
        assert initial_state == {'source_uuid': 'result-source'}
        return {
            'source_facts': [1, 2, 3],
            'triplet_occurrences': [1, 2],
            'source_entity_description_persisted_count': 3,
            'source_event_description_persisted_count': 4,
            'source_predicate_description_persisted_count': 5,
            'source_statement_description_persisted_count': 6,
            'source_procedure_description_persisted_count': 7,
            'source_entity_hub_count': 8,
            'source_event_hub_count': 9,
            'source_triplet_hub_count': 11,
            'source_predicate_hub_count': 10,
            'source_statement_hub_count': 11,
            'source_procedure_hub_count': 12,
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


def test_ingest_does_not_report_success_when_ownership_attach_fails(
    monkeypatch,
):
    settings = Settings()
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


def test_ingest_source_reports_all_final_state_counts(monkeypatch):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    graph = _SourceProcessingGraph()
    composed = _ComposedGraph(graph)
    composition_calls = []
    schema_calls = []

    async def ensure_schema(session_factory, *, embedding_dimension):
        schema_calls.append((session_factory, embedding_dimension))

    monkeypatch.setattr(application.schema, 'ensure_schema', ensure_schema)

    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    def compose(
        received_settings,
        received_runtime,
        received_database,
        *,
        tokenizers,
        recorder,
    ):
        composition_calls.append(
            (
                received_settings,
                received_runtime,
                received_database,
                recorder,
            )
        )
        return composed

    monkeypatch.setattr(application, 'build_source_processing_graph', compose)

    result = asyncio.run(
        application.ingest_source(
            settings,
            'user-1',
            'fixtures/book.pdf',
            [0, 2],
        )
    )

    assert runtime.entered is True
    assert composition_calls == [(settings, runtime, database, None)]
    assert database.closed is True
    assert composed.build_calls == 1
    assert graph.initial_state == {
        'pdf_path': 'fixtures/book.pdf',
        'pages': [0, 2],
        'source': graph.initial_state['source'],
    }
    assert len(schema_calls) == 1
    assert schema_calls[0][1] == settings.local_models.embedding.model.dimension
    assert graph.initial_state['source'].key == 'book.pdf'
    assert result.source == Source(uuid='result-source', key='book.pdf')
    assert result.ocr_page_count == 1
    assert result.corrected_page_count == 2
    assert result.formatted_page_count == 3
    assert result.text_seam_page_count == 1
    assert result.image_seam_page_count == 2
    assert result.image_description_page_count == 1
    assert result.split_page_count == 3
    assert result.instruction_count == 2
    assert result.exercise_component_count == 1
    assert result.pedagogical_component_count == 2
    assert result.statement_count == 1
    assert result.procedure_count == 3
    assert result.embedded_page_count == 2


def test_ingest_source_closes_database_when_graph_fails(monkeypatch):
    settings = Settings()
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


def test_ingest_source_passes_one_opt_in_recorder(monkeypatch, tmp_path):
    settings = Settings(
        training=TrainingSettings(examples_directory=tmp_path),
    )
    database = _Database(settings.database)
    runtime = _Runtime(settings.local_models)
    graph = _SourceProcessingGraph()
    recorder_calls = []

    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    def compose(*args, tokenizers, recorder):
        recorder_calls.append(recorder)
        return _ComposedGraph(graph)

    monkeypatch.setattr(application, 'build_source_processing_graph', compose)

    asyncio.run(
        application.ingest_source(settings, 'user-1', 'fixtures/book.pdf')
    )

    assert len(recorder_calls) == 1
    assert isinstance(recorder_calls[0], application.Recorder)
    assert recorder_calls[0]._directory == tmp_path
    assert recorder_calls[0]._run_directory.parent == tmp_path
    assert recorder_calls[0]._run_directory.name.endswith('Z')


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


def test_run_source_semantic_stage_uses_one_complete_semantic_graph(
    monkeypatch,
):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    graph = _SourceSemanticGraph()
    composed = _ComposedGraph(graph)
    composition_calls = []

    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    def compose(
        received_settings,
        received_runtime,
        received_database,
        *,
        tokenizers,
        recorder,
    ):
        composition_calls.append(
            (
                received_settings,
                received_runtime,
                received_database,
                recorder,
            )
        )
        return composed

    monkeypatch.setattr(application, 'build_source_semantic_graph', compose)

    result = asyncio.run(
        application.run_source_semantic_stage(
            settings,
            'user-1',
            'result-source',
        )
    )

    assert result.triplet_count == 2
    assert result.source_fact_count == 3
    assert result.source_entity_hub_count == 8
    assert result.source_event_hub_count == 9
    assert result.source_triplet_hub_count == 11
    assert result.source_predicate_hub_count == 10
    assert result.source_statement_hub_count == 11
    assert result.source_procedure_hub_count == 12
    assert result.source_entity_description_count == 3
    assert result.source_event_description_count == 4
    assert result.source_predicate_description_count == 5
    assert result.source_statement_description_count == 6
    assert result.source_procedure_description_count == 7
    assert composition_calls == [(settings, runtime, database, None)]
    assert database.closed is True


def test_combined_pipeline_reuses_resources_and_recorder(monkeypatch, tmp_path):
    settings = Settings(
        training=TrainingSettings(examples_directory=tmp_path),
    )
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    source_graph = _SourceProcessingGraph()
    semantic_graph = _SourceSemanticGraph()
    source_composed = _ComposedGraph(source_graph)
    semantic_composed = _ComposedGraph(semantic_graph)
    source_calls = []
    semantic_calls = []
    schema_calls = []

    async def ensure_schema(session_factory, *, embedding_dimension):
        schema_calls.append((session_factory, embedding_dimension))

    monkeypatch.setattr(application.schema, 'ensure_schema', ensure_schema)

    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    def compose_source(*args, tokenizers, recorder):
        source_calls.append((*args, recorder))
        return source_composed

    def compose_semantic(*args, tokenizers, recorder):
        semantic_calls.append((*args, recorder))
        return semantic_composed

    monkeypatch.setattr(
        application, 'build_source_processing_graph', compose_source
    )
    monkeypatch.setattr(
        application, 'build_source_semantic_graph', compose_semantic
    )

    result = asyncio.run(
        application.ingest_and_run_source_semantic_stage(
            settings,
            'user-1',
            'fixtures/book.pdf',
        )
    )
    assert len(schema_calls) == 1
    assert schema_calls[0][1] == settings.local_models.embedding.model.dimension

    assert result.source_processing_stage.source.uuid == 'result-source'
    assert result.source_semantic_stage.triplet_count == 2
    assert result.source_semantic_stage.source_fact_count == 3
    assert source_calls[0][1:3] == (runtime, database)
    assert semantic_calls[0][1:3] == (runtime, database)
    assert source_calls[0][3] is semantic_calls[0][3]
    assert runtime.entered is True
    assert database.closed is True


def test_schema_failure_closes_database_before_starting_runtime(monkeypatch):
    settings = Settings()
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


def test_ingest_source_cancellation_closes_runtime_before_database(monkeypatch):
    settings = Settings()
    events = []

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
    monkeypatch.setattr(
        application,
        'build_source_processing_graph',
        lambda *args, **kwargs: _ComposedGraph(BlockingGraph()),
    )

    async def exercise():
        task = asyncio.create_task(
            application.ingest_source(settings, 'user-1', 'fixtures/book.pdf')
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert events == ['runtime-close', 'database-close']


def test_run_global_semantic_stage_reports_global_hub_count(monkeypatch):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)

    class _GlobalSemanticGraph:
        async def ainvoke(self, initial_state):
            assert initial_state == {}
            return {
                'global_entity_hub_count': 1,
                'global_event_hub_count': 2,
                'global_predicate_hub_count': 4,
                'global_triplet_count': 6,
                'global_triplet_hub_count': 7,
                'global_statement_hub_count': 3,
                'global_procedure_hub_count': 1,
            }

    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda _, **kwargs: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application,
        'build_global_semantic_graph',
        lambda *args, **kwargs: _ComposedGraph(_GlobalSemanticGraph()),
    )

    result = asyncio.run(application.run_global_semantic_stage(settings))

    assert result.global_entity_hub_count == 1
    assert result.global_event_hub_count == 2
    assert result.global_predicate_hub_count == 4
    assert result.global_triplet_count == 6
    assert result.global_triplet_hub_count == 7
    assert result.global_statement_hub_count == 3
    assert result.global_procedure_hub_count == 1


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
