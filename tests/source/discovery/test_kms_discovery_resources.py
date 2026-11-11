"""Tests for invocation-owned discovery HTTP resources."""

import asyncio
from typing import Any

import pytest

from kms.source.discovery import resources
from kms.source.discovery.settings import (
    SourceDiscoverySettings,
    TavilySettings,
)


class _TrackedClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


class _SDK:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _settings() -> SourceDiscoverySettings:
    return SourceDiscoverySettings(
        tavily=TavilySettings(api_key='test-key', timeout_seconds=12.0)
    )


@pytest.mark.anyio
async def test_discovery_resources_own_only_the_tavily_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_TrackedClient] = []
    sdks: list[_SDK] = []

    def create_client(**kwargs: Any) -> _TrackedClient:
        client = _TrackedClient(**kwargs)
        clients.append(client)
        return client

    def create_sdk(**kwargs: Any) -> _SDK:
        sdk = _SDK(**kwargs)
        sdks.append(sdk)
        return sdk

    monkeypatch.setattr(resources.httpx, 'AsyncClient', create_client)
    monkeypatch.setattr(resources, 'AsyncTavilyClient', create_sdk)

    async with resources.discovery_resources(_settings()):
        assert clients[0].closed is False

    assert len(clients) == 1
    assert clients[0].kwargs == {
        'base_url': 'https://api.tavily.com',
        'timeout': 12.0,
        'trust_env': False,
    }
    assert clients[0].closed is True
    assert sdks[0].kwargs['api_key'] == 'test-key'
    assert sdks[0].kwargs['client'] is clients[0]


@pytest.mark.anyio
async def test_discovery_resources_close_client_on_sdk_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_TrackedClient] = []

    def create_client(**kwargs: Any) -> _TrackedClient:
        client = _TrackedClient(**kwargs)
        clients.append(client)
        return client

    def fail_sdk(**kwargs: Any) -> None:
        raise RuntimeError('SDK setup failed')

    monkeypatch.setattr(resources.httpx, 'AsyncClient', create_client)
    monkeypatch.setattr(resources, 'AsyncTavilyClient', fail_sdk)

    with pytest.raises(RuntimeError, match='SDK setup failed'):
        async with resources.discovery_resources(_settings()):
            pytest.fail('setup should fail before yielding')

    assert clients[0].closed is True


@pytest.mark.anyio
async def test_discovery_resources_close_client_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_TrackedClient] = []

    def create_client(**kwargs: Any) -> _TrackedClient:
        client = _TrackedClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(resources.httpx, 'AsyncClient', create_client)
    monkeypatch.setattr(resources, 'AsyncTavilyClient', _SDK)
    entered = asyncio.Event()

    async def use_resources() -> None:
        async with resources.discovery_resources(_settings()):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(use_resources())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert clients[0].closed is True
