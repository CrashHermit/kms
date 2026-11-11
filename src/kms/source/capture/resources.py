"""Invocation-owned URL-inspection and rendering clients."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from kms.source.capture.capture import SourceCapture
from kms.source.capture.inspection import PublicURLInspector
from kms.source.capture.rendering import GotenbergRenderer
from kms.source.capture.settings import SourceCaptureSettings


@asynccontextmanager
async def capture_resources(
    settings: SourceCaptureSettings,
) -> AsyncIterator[SourceCapture]:
    """Own the two external HTTP clients used by one capture invocation."""
    clients: list[httpx.AsyncClient] = []
    try:
        inspector_http = httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        clients.append(inspector_http)
        renderer_http = httpx.AsyncClient(
            base_url=settings.gotenberg_url,
            timeout=settings.timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )
        clients.append(renderer_http)
        yield SourceCapture(
            PublicURLInspector(
                inspector_http, max_redirects=settings.max_redirects
            ),
            GotenbergRenderer(renderer_http),
        )
    finally:

        async def close_clients() -> None:
            await asyncio.gather(*(client.aclose() for client in clients))

        close_task = asyncio.create_task(close_clients())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await close_task
            raise
