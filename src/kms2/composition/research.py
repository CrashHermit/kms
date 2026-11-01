"""Invocation-owned HTTP resources for external research."""

import asyncio
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from tavily import AsyncTavilyClient

from kms2.config.settings import Settings
from kms2.retrieval.download import PublicMediaClient
from kms2.retrieval.tavily import TavilyWebClient


@dataclass(frozen=True)
class ResearchResources:
    """Invocation-bound retrieval clients and the temporary document directory."""

    tavily: TavilyWebClient
    media: PublicMediaClient
    temporary_directory: Path


@asynccontextmanager
async def research_resources(
    settings: Settings,
) -> AsyncIterator[ResearchResources]:
    """Own separate provider/media clients and temporary data for one invocation."""
    research_settings = settings.research
    key = research_settings.tavily.api_key
    api_key = key.get_secret_value().strip() if key is not None else ''
    if research_settings.enabled and not api_key:
        raise ValueError(
            'KMS2_RESEARCH__TAVILY__API_KEY is required when research is enabled'
        )

    temporary_directory = tempfile.TemporaryDirectory(prefix='kms2-research-')
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
            temporary_directory=Path(temporary_directory.name),
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
        finally:
            temporary_directory.cleanup()
