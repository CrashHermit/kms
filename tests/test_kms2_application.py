import asyncio

from kms2 import application
from kms2.config import Settings
from kms2.core.model import Source


class _Runtime:
    def __init__(self, settings):
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

    async def close(self):
        self.closed = True

    def session(self):
        return 'session'


class _Graph:
    def __init__(self, fail=False, events=None):
        self.initial_state = None
        self.fail = fail
        self.events = events

    async def ainvoke(self, initial_state):
        self.initial_state = initial_state
        if self.events is not None:
            self.events.append('source-invoke')
        if self.fail:
            raise RuntimeError('graph failed')
        return {
            'source': Source(uuid='result-source', key='book.pdf'),
            'split_pages': ['page-1', 'page-2', 'page-3'],
        }


class _SemanticGraph:
    def __init__(self, fail=False, events=None):
        self.initial_state = None
        self.fail = fail
        self.events = events

    async def ainvoke(self, initial_state):
        self.initial_state = initial_state
        if self.events is not None:
            self.events.append('semantic-invoke')
        if self.fail:
            raise RuntimeError('semantic graph failed')
        return {'raw_assertions': ['triplet-1', 'triplet-2']}


class _ComposedGraph:
    def __init__(self, graph):
        self.graph = graph
        self.build_calls = 0

    def build_graph(self, raw_only=False):
        self.build_calls += 1
        return self.graph


def test_ingest_source_owns_runtime_and_invokes_composed_graph(monkeypatch):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    graph = _Graph()
    composed = _ComposedGraph(graph)
    composition_calls = []

    def make_runtime(runtime_settings):
        assert runtime_settings is settings.local_models
        return runtime

    def make_database(database_settings):
        assert database_settings is settings.database
        return database

    def compose(received_settings, received_runtime, received_database):
        composition_calls.append(
            (received_settings, received_runtime, received_database)
        )
        return composed

    monkeypatch.setattr(application, 'LocalModelRuntime', make_runtime)
    monkeypatch.setattr(application, 'DatabaseClient', make_database)
    monkeypatch.setattr(application, 'build_source_graph', compose)

    result = asyncio.run(
        application.ingest_source(settings, 'fixtures/book.pdf', [0, 2])
    )

    assert runtime.entered is True
    assert composition_calls == [(settings, runtime, database)]
    assert database.closed is True
    assert composed.build_calls == 1
    assert graph.initial_state == {
        'pdf_path': 'fixtures/book.pdf',
        'pages': [0, 2],
        'source': graph.initial_state['source'],
    }
    assert graph.initial_state['source'].key == 'book.pdf'
    assert result.source == Source(uuid='result-source', key='book.pdf')
    assert result.page_count == 3


def test_ingest_source_closes_database_when_graph_fails(monkeypatch):
    settings = Settings()
    database = _Database(settings.database)
    graph = _Graph(fail=True)

    monkeypatch.setattr(application, 'LocalModelRuntime', _Runtime)
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application,
        'build_source_graph',
        lambda *_: _ComposedGraph(graph),
    )

    try:
        asyncio.run(application.ingest_source(settings, 'fixtures/book.pdf'))
    except RuntimeError as error:
        assert str(error) == 'graph failed'
    else:
        raise AssertionError('graph failure was not propagated')

    assert database.closed is True


def test_ingest_and_extract_triplets_reuses_resources_and_orders_graphs(
    monkeypatch,
):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    events = []
    source_graph = _Graph(events=events)
    semantic_graph = _SemanticGraph(events=events)
    source_composed = _ComposedGraph(source_graph)
    semantic_composed = _ComposedGraph(semantic_graph)
    runtime_calls = []
    database_calls = []

    def make_runtime(runtime_settings):
        runtime_calls.append(runtime_settings)
        return runtime

    def make_database(database_settings):
        database_calls.append(database_settings)
        return database

    def compose_source(received_settings, received_runtime, received_database):
        assert (received_settings, received_runtime, received_database) == (
            settings,
            runtime,
            database,
        )
        events.append('source-compose')
        return source_composed

    def compose_semantic(
        received_settings, received_runtime, received_database
    ):
        assert (received_settings, received_runtime, received_database) == (
            settings,
            runtime,
            database,
        )
        events.append('semantic-compose')
        return semantic_composed

    monkeypatch.setattr(application, 'LocalModelRuntime', make_runtime)
    monkeypatch.setattr(application, 'DatabaseClient', make_database)
    monkeypatch.setattr(application, 'build_source_graph', compose_source)
    monkeypatch.setattr(application, 'build_semantic_graph', compose_semantic)

    result = asyncio.run(
        application.ingest_and_extract_triplets(
            settings,
            'fixtures/book.pdf',
            [0, 2],
        )
    )

    assert runtime_calls == [settings.local_models]
    assert runtime.entered is True
    assert database_calls == [settings.database]
    assert database.closed is True
    assert source_composed.build_calls == 1
    assert semantic_composed.build_calls == 1
    assert events == [
        'source-compose',
        'source-invoke',
        'semantic-compose',
        'semantic-invoke',
    ]
    assert semantic_graph.initial_state == {'source_uuid': 'result-source'}
    assert result.source == Source(uuid='result-source', key='book.pdf')
    assert result.page_count == 3
    assert result.triplet_count == 2


def test_ingest_and_extract_triplets_closes_database_when_extraction_fails(
    monkeypatch,
):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    source_graph = _Graph()
    semantic_graph = _SemanticGraph(fail=True)

    monkeypatch.setattr(application, 'LocalModelRuntime', lambda _: runtime)
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application,
        'build_source_graph',
        lambda *_: _ComposedGraph(source_graph),
    )
    monkeypatch.setattr(
        application,
        'build_semantic_graph',
        lambda *_: _ComposedGraph(semantic_graph),
    )

    try:
        asyncio.run(
            application.ingest_and_extract_triplets(
                settings,
                'fixtures/book.pdf',
            )
        )
    except RuntimeError as error:
        assert str(error) == 'semantic graph failed'
    else:
        raise AssertionError('semantic graph failure was not propagated')

    assert database.closed is True


def test_list_sources_closes_database_and_returns_sources(monkeypatch):
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

        async def list_sources(self):
            return sources

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(application, 'SourceRepository', _SourceRepository)

    result = asyncio.run(application.list_sources(settings))

    assert result == sources
    assert len(repository_calls) == 1
    assert database.closed is True


class _CompleteSemanticGraph:
    def __init__(self, events):
        self.events = events

    def build_graph(self):
        self.events.append('semantic-compose')
        return self

    async def ainvoke(self, initial_state):
        self.events.append('semantic-invoke')
        assert initial_state == {'source_uuid': 'source-1'}
        return {
            'raw_assertions': [1, 2],
            'source_entity_description_persisted_count': 3,
            'source_event_description_persisted_count': 4,
            'source_predicate_description_persisted_count': 5,
        }


def test_run_semantic_stage_uses_one_complete_semantic_graph(monkeypatch):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)
    events = []
    graph = _CompleteSemanticGraph(events)

    monkeypatch.setattr(application, 'LocalModelRuntime', lambda _: runtime)
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application,
        'build_semantic_graph',
        lambda received_settings, received_runtime, received_database: graph,
    )

    result = asyncio.run(application.run_semantic_stage(settings, 'source-1'))

    assert result.raw_assertion_count == 2
    assert result.source_entity_description_count == 3
    assert result.source_event_description_count == 4
    assert result.source_predicate_description_count == 5
    assert events == ['semantic-compose', 'semantic-invoke']
    assert database.closed is True
