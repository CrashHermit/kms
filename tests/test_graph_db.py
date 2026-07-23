"""Connection plumbing for the graph tier — pure, no server (neo4j is stubbed in conftest).
Covers the config helpers and the driver singleton's lifecycle, not any graph model."""

import asyncio
import json

import httpx
import pytest

from kms.graph.db import (
    HttpQueryDriver,
    HttpQueryError,
    _http_base_url,
    _use_http,
    close_driver,
    database,
    driver,
    is_configured,
)

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


# --- HTTP Query-API transport: selection, URL derivation, and request/response shaping ---


def test_transport_env_selects_http_over_a_bolt_uri(monkeypatch):
    # An Aura Bolt URI + an explicit http transport -> HTTP, with the HTTPS endpoint derived
    # from the same host (the Bolt-blocked-sandbox path: flip one env var, keep the URI).
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://abc123.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_TRANSPORT", "http")
    assert _use_http() is True
    assert _http_base_url() == "https://abc123.databases.neo4j.io"


def test_transport_is_inferred_from_an_https_uri_scheme(monkeypatch):
    monkeypatch.delenv("NEO4J_TRANSPORT", raising=False)
    monkeypatch.setenv("NEO4J_URI", "https://abc123.databases.neo4j.io")
    assert _use_http() is True
    assert _http_base_url() == "https://abc123.databases.neo4j.io"


def test_bolt_stays_the_default_transport(monkeypatch):
    monkeypatch.delenv("NEO4J_TRANSPORT", raising=False)
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://abc123.databases.neo4j.io")
    assert _use_http() is False


def test_driver_returns_the_http_facade_when_selected(monkeypatch):
    _set_conn(monkeypatch)
    monkeypatch.setenv("NEO4J_TRANSPORT", "http")
    asyncio.run(close_driver())
    try:
        assert isinstance(driver(), HttpQueryDriver)
    finally:
        asyncio.run(close_driver())


def _mock_driver(handler) -> HttpQueryDriver:
    """An HttpQueryDriver whose client answers from a handler instead of the network, so the
    request-shaping and response-parsing are testable with no server."""
    d = HttpQueryDriver("https://host.databases.neo4j.io", ("neo4j", "pw"))
    d._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return d


def test_http_run_posts_to_query_v2_and_parses_records():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"fields": ["ok"], "values": [[1]]}})

    d = _mock_driver(handler)

    async def scenario():
        try:
            async with d.session(database="neo4j") as s:
                result = await s.run("RETURN 1 AS ok", answer=42)
                record = await result.single()
            assert record["ok"] == 1
        finally:
            await d.close()

    asyncio.run(scenario())
    assert seen["url"] == "https://host.databases.neo4j.io/db/neo4j/query/v2"
    assert seen["body"] == {"statement": "RETURN 1 AS ok", "parameters": {"answer": 42}}


def test_http_run_raises_neo4j_error_with_its_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "errors": [
                    {
                        "code": "Neo.ClientError.Security.Unauthorized",
                        "message": "Invalid credential.",
                    }
                ]
            },
        )

    d = _mock_driver(handler)

    async def scenario():
        try:
            async with d.session(database="neo4j") as s:
                await s.run("RETURN 1")
        finally:
            await d.close()

    with pytest.raises(HttpQueryError, match="Unauthorized"):
        asyncio.run(scenario())


def test_http_verify_connectivity_round_trips():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"fields": ["ok"], "values": [[1]]}})

    d = _mock_driver(handler)

    async def scenario():
        try:
            await d.verify_connectivity()  # must not raise on a healthy RETURN 1
        finally:
            await d.close()

    asyncio.run(scenario())
