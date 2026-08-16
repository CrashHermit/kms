import asyncio
from types import SimpleNamespace

from kms import runtime


def test_runtime_owns_process_services(monkeypatch):
    settings = SimpleNamespace(serving=SimpleNamespace(manage=False))
    session_factory = object()
    closed = []

    monkeypatch.setattr(runtime.config, 'get_settings', lambda: settings)
    monkeypatch.setattr(runtime.db, 'is_configured', lambda: True)
    monkeypatch.setattr(runtime.db, 'session', session_factory)

    async def close_driver():
        closed.append(True)

    monkeypatch.setattr(runtime.db, 'close_driver', close_driver)
    application = runtime.Runtime()

    assert application.neo4j_configured is True
    assert application.session_factory() is session_factory
    assert application.model_manager is None

    asyncio.run(application.close())
    assert closed == [True]
