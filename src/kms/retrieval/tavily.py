"""Normalized Tavily search client."""

from tavily import AsyncTavilyClient

from kms.config.research import TavilySettings
from kms.core.model.research.retrieval import WebSearchHit


class TavilyWebClient:
    """Search the web through Tavily's async SDK."""

    def __init__(
        self,
        settings: TavilySettings,
        client: AsyncTavilyClient,
    ) -> None:
        self._settings = settings
        self._client = client

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
            )
            for result in response.get('results', [])
        ]
