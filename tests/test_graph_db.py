import asyncio

import neo4j
import pytest

from kms.graph import db

_CONN_ENV = {
    'KMS_DATABASE__URI': 'bolt://localhost:7687',
    'KMS_DATABASE__USERNAME': 'neo4j',
    'KMS_DATABASE__PASSWORD': 'secret',
}


def _set_conn(monkeypatch):
    for k, v in _CONN_ENV.items():
        monkeypatch.setenv(k, v)


def test_is_configured_tracks_the_uri_env(monkeypatch):
    monkeypatch.delenv('KMS_DATABASE__URI', raising=False)
    assert db.is_configured() is False
    monkeypatch.setenv('KMS_DATABASE__URI', 'bolt://localhost:7687')
    assert db.is_configured() is True


def test_database_defaults_to_neo4j_and_honours_override(monkeypatch):
    monkeypatch.delenv('KMS_DATABASE__DATABASE', raising=False)
    assert db.database() == 'neo4j'
    monkeypatch.setenv('KMS_DATABASE__DATABASE', 'kms')
    assert db.database() == 'kms'


def test_driver_raises_a_clear_error_when_unconfigured(monkeypatch):
    for k in _CONN_ENV:
        monkeypatch.delenv(k, raising=False)
    asyncio.run(db.close_driver())
    with pytest.raises(RuntimeError, match='KMS_DATABASE__URI is not set'):
        db.driver()


def test_driver_is_a_reused_singleton_until_closed(monkeypatch):
    _set_conn(monkeypatch)
    try:
        first = db.driver()
        assert db.driver() is first
    finally:
        asyncio.run(db.close_driver())


def test_driver_is_the_native_bolt_driver(monkeypatch):
    _set_conn(monkeypatch)
    asyncio.run(db.close_driver())
    try:
        assert isinstance(db.driver(), neo4j.AsyncDriver)
    finally:
        asyncio.run(db.close_driver())


def test_close_driver_is_a_safe_noop_when_nothing_opened(monkeypatch):
    _set_conn(monkeypatch)
    asyncio.run(db.close_driver())
    opened = db.driver()
    asyncio.run(db.close_driver())
    assert db.driver() is not opened
    asyncio.run(db.close_driver())
