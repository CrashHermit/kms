import asyncio

from kms2 import application
from kms2.config import Settings
from kms2.core.model.source import Source


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


class _Graph:
    def __init__(self, fail=False):
        self.initial_state = None
        self.fail = fail

    async def ainvoke(self, initial_state):
        self.initial_state = initial_state
        if self.fail:
            raise RuntimeError('graph failed')
        return {
            'source': Source(uuid='result-source', key='book.pdf'),
            'split_pages': ['page-1', 'page-2', 'page-3'],
        }


class _ComposedGraph:
    def __init__(self, graph):
        self.graph = graph
        self.build_calls = 0

    def build_graph(self):
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
