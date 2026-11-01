"""Normalized Tavily search and extraction client."""

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from tavily import AsyncTavilyClient

from kms2.config.research import TavilySettings
from kms2.core.model.research.retrieval import (
    WebExtraction,
    WebExtractionBatch,
    WebExtractionFailure,
    WebSearchHit,
)
from kms2.retrieval.download import validate_public_url

_DnsResolver = Callable[[str, int], Awaitable[list[str]]]


class TavilyWebClient:
    """Search the web and extract selected URLs through Tavily's async SDK."""

    def __init__(
        self,
        settings: TavilySettings,
        client: AsyncTavilyClient,
        *,
        resolver: _DnsResolver | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._resolver = resolver

    async def search(self, query: str) -> list[WebSearchHit]:
        """Return Tavily search hits without generated answers or raw pages."""
        response = await self._client.search(
            query,
            search_depth=self._settings.search_depth,
            max_results=self._settings.result_limit,
            include_answer=False,
            include_raw_content=False,
            auto_parameters=False,
            timeout=self._settings.timeout_seconds,
        )
        return [
            WebSearchHit(
                title=result.get('title'),
                url=result['url'],
                snippet=result.get('content', ''),
                score=result['score'],
            )
            for result in response.get('results', [])
        ]

    async def extract(self, urls: list[str]) -> WebExtractionBatch:
        """Extract unique, publicly hosted URLs and retain URL-specific failures."""
        unique_urls = list(dict.fromkeys(urls))
        if not unique_urls:
            return WebExtractionBatch()

        for url in unique_urls:
            await validate_public_url(url, resolver=self._resolver)

        extractions: dict[str, WebExtraction] = {}
        failures: dict[str, WebExtractionFailure] = {}
        for offset in range(0, len(unique_urls), 20):
            group = unique_urls[offset : offset + 20]
            response = await self._client.extract(
                group,
                format='markdown',
                extract_depth=self._settings.extract_depth,
                include_images=True,
                timeout=self._settings.timeout_seconds,
            )
            received_at = datetime.now(UTC)
            requested_group = set(group)
            for result in response.get('results', []):
                url = result['url']
                if url in requested_group:
                    extractions[url] = WebExtraction(
                        url=url,
                        retrieved_at=received_at,
                        raw_markdown=result.get('raw_content', '') or '',
                        image_urls=_image_urls(result.get('images', [])),
                    )
            for failure in response.get('failed_results', []):
                url = failure['url']
                if url in requested_group:
                    failures[url] = WebExtractionFailure(
                        url=url,
                        error=failure.get('error', ''),
                    )

        return WebExtractionBatch(
            results=[
                extractions[url] for url in unique_urls if url in extractions
            ],
            failures=[failures[url] for url in unique_urls if url in failures],
        )


def _image_urls(images: Sequence[Any]) -> list[str]:
    """Normalize the SDK's image URL entries without inventing destinations."""
    urls: list[str] = []
    for image in images:
        if isinstance(image, str):
            urls.append(image)
        elif isinstance(image, dict) and isinstance(image.get('url'), str):
            urls.append(image['url'])
    return urls
