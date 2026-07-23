"""Connection plumbing for the graph tier — pure, no server (neo4j is stubbed in conftest).
Covers the config helpers and the driver singleton's lifecycle, not any graph model."""

import asyncio

import pytest

from kms.graph.db import close_driver, database, driver, is_configured

_CONN_ENV = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "secret",
}


def _set_conn(monkeypatch):
    for k, v in _CONN_ENV.items():
        monkeypatch.setenv(k, v)


def test_is_configured_tracks_the_uri_env(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    assert is_configured() is False
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    assert is_configured() is True


def test_database_defaults_to_neo4j_and_honours_override(monkeypatch):
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    assert database() == "neo4j"
    monkeypatch.setenv("NEO4J_DATABASE", "kms")
    assert database() == "kms"


def test_driver_raises_a_clear_error_when_unconfigured(monkeypatch):
    for k in _CONN_ENV:
        monkeypatch.delenv(k, raising=False)
    asyncio.run(close_driver())  # ensure no singleton lingers from another test
    with pytest.raises(RuntimeError, match="NEO4J_URI is not set"):
        driver()


def test_driver_is_a_reused_singleton_until_closed(monkeypatch):
    _set_conn(monkeypatch)
    try:
        first = driver()
        assert driver() is first  # same pooled instance, not rebuilt per call
    finally:
        asyncio.run(close_driver())


def test_close_driver_is_a_safe_noop_when_nothing_opened(monkeypatch):
    _set_conn(monkeypatch)
    asyncio.run(close_driver())  # never opened -> must not raise
    opened = driver()
    asyncio.run(close_driver())  # closing a real one resets the singleton
    assert driver() is not opened  # a fresh instance after close
    asyncio.run(close_driver())
