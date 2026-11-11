"""Invocation-owned Tavily HTTP resources."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from tavily import AsyncTavilyClient

from kms.source.discovery.search import TavilyWebClient
from kms.source.discovery.settings import SourceDiscoverySettings


@asynccontextmanager
async def discovery_resources(
    settings: SourceDiscoverySettings,
) -> AsyncIterator[TavilyWebClient]:
    """Own the Tavily HTTP client for one discovery invocation."""
    tavily_settings = settings.tavily
    clients: list[httpx.AsyncClient] = []
    try:
        tavily_http = httpx.AsyncClient(
            base_url='https://api.tavily.com',
            timeout=tavily_settings.timeout_seconds,
            trust_env=False,
        )
        clients.append(tavily_http)
        tavily_sdk = AsyncTavilyClient(
            api_key=tavily_settings.api_key.get_secret_value(),
            api_base_url='https://api.tavily.com',
            client=tavily_http,
        )
        yield TavilyWebClient(tavily_settings, tavily_sdk)
    finally:

        async def close_clients() -> None:
            await asyncio.gather(*(client.aclose() for client in clients))

        close_task = asyncio.create_task(close_clients())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await close_task
            raise
