"""Lifecycle tests for capture-owned HTTP clients."""

import asyncio

import pytest

from kms.source.capture import resources
from kms.source.capture.capture import SourceCapture
from kms.source.capture.settings import SourceCaptureSettings


class _TrackedClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_capture_resources_own_two_configured_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        clients: list[_TrackedClient] = []

        def create_client(**kwargs: object) -> _TrackedClient:
            client = _TrackedClient(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(resources.httpx, 'AsyncClient', create_client)
        settings = SourceCaptureSettings(
            gotenberg_url='http://renderer.local:3000',
            timeout_seconds=12.0,
            max_redirects=3,
        )
        async with resources.capture_resources(settings) as capture:
            assert isinstance(capture, SourceCapture)
            assert len(clients) == 2
            assert clients[0].kwargs == {
                'timeout': 12.0,
                'trust_env': False,
                'follow_redirects': False,
                'limits': resources.httpx.Limits(max_keepalive_connections=0),
            }
            assert clients[1].kwargs == {
                'base_url': 'http://renderer.local:3000',
                'timeout': 12.0,
                'trust_env': False,
                'follow_redirects': False,
            }
        assert all(client.closed for client in clients)

    asyncio.run(exercise())


def test_capture_resources_close_partial_client_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        clients: list[_TrackedClient] = []

        def create_client(**kwargs: object) -> _TrackedClient:
            if clients:
                raise RuntimeError('renderer client setup failed')
            client = _TrackedClient(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(resources.httpx, 'AsyncClient', create_client)
        with pytest.raises(RuntimeError, match='renderer client setup failed'):
            async with resources.capture_resources(SourceCaptureSettings()):
                pytest.fail('resource setup should fail before yielding')
        assert clients[0].closed

    asyncio.run(exercise())


def test_capture_resources_close_clients_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        clients: list[_TrackedClient] = []

        def create_client(**kwargs: object) -> _TrackedClient:
            client = _TrackedClient(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(resources.httpx, 'AsyncClient', create_client)
        entered = asyncio.Event()

        async def capture() -> None:
            async with resources.capture_resources(SourceCaptureSettings()):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(capture())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(clients) == 2
        assert all(client.closed for client in clients)

    asyncio.run(exercise())
