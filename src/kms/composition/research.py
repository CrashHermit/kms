"""Invocation-owned HTTP resources for external research."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from tavily import AsyncTavilyClient

from kms.config.settings import Settings
from kms.retrieval.download import PublicMediaClient
from kms.retrieval.tavily import TavilyWebClient


@dataclass(frozen=True)
class ResearchResources:
    """Invocation-bound Tavily, media, and screenshot clients."""

    tavily: TavilyWebClient
    media: PublicMediaClient
    gotenberg: httpx.AsyncClient


@asynccontextmanager
async def research_resources(
    settings: Settings,
) -> AsyncIterator[ResearchResources]:
    """Own separate provider and media clients for one invocation."""
    research_settings = settings.research
    key = research_settings.tavily.api_key
    api_key = key.get_secret_value().strip() if key is not None else ''
    if research_settings.enabled and not api_key:
        raise ValueError(
            'KMS_RESEARCH__TAVILY__API_KEY is required when research is enabled'
        )

    clients: list[httpx.AsyncClient] = []
    try:
        tavily_http = httpx.AsyncClient(
            base_url='https://api.tavily.com',
            timeout=research_settings.tavily.timeout_seconds,
            trust_env=False,
        )
        clients.append(tavily_http)
        media_http = httpx.AsyncClient(
            timeout=research_settings.download_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        clients.append(media_http)
        gotenberg_http = httpx.AsyncClient(
            base_url=research_settings.gotenberg_url,
            timeout=research_settings.download_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )
        clients.append(gotenberg_http)
        tavily_sdk = AsyncTavilyClient(
            api_key=api_key,
            api_base_url='https://api.tavily.com',
            client=tavily_http,
        )
        yield ResearchResources(
            tavily=TavilyWebClient(research_settings.tavily, tavily_sdk),
            media=PublicMediaClient(
                media_http,
                max_download_bytes=research_settings.max_download_bytes,
                max_redirects=research_settings.max_redirects,
            ),
            gotenberg=gotenberg_http,
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
