"""Tavily-backed source discovery search adapter."""

from tavily import AsyncTavilyClient

from kms.source.discovery.models import SourceCandidate
from kms.source.discovery.settings import TavilySettings


class TavilyWebClient:
    """Search the web through Tavily's async SDK."""

    def __init__(
        self,
        settings: TavilySettings,
        client: AsyncTavilyClient,
    ) -> None:
        self._settings = settings
        self._client = client

    async def search(self, query: str) -> list[SourceCandidate]:
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
            SourceCandidate(
                title=result['title'],
                url=result['url'],
                snippet=result['content'],
            )
            for result in response['results']
        ]
