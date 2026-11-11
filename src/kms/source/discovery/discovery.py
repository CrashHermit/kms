"""Search and filter source candidates without acquiring or storing them."""

from kms.core.models import Position
from kms.source.discovery.models import SourceCandidate
from kms.source.discovery.search import TavilyWebClient
from kms.source.discovery.selection import (
    DiscoverySearchResult,
    DiscoverySelectionInput,
    DiscoverySelectionModule,
)


class SourceDiscovery:
    """Discover relevant candidates from one Tavily search."""

    def __init__(
        self,
        web: TavilyWebClient,
        selector: DiscoverySelectionModule,
    ) -> None:
        self._web = web
        self._selector = selector

    async def discover(self, question: str) -> list[SourceCandidate]:
        """Search once and return selected candidates in provider order."""
        hits = await self._web.search(question)
        offered: list[DiscoverySearchResult] = []
        seen_urls: set[str] = set()
        candidates: list[SourceCandidate] = []
        for hit in hits:
            if hit.url in seen_urls:
                continue
            seen_urls.add(hit.url)
            offered.append(
                DiscoverySearchResult(
                    position=Position(len(offered)),
                    title=hit.title,
                    url=hit.url,
                    snippet=hit.snippet,
                )
            )
            candidates.append(hit)

        if not offered:
            return []

        selection = await self._selector.acall(
            request=DiscoverySelectionInput(question=question, results=offered)
        )
        selected = set(selection.result_positions)
        return [
            candidate
            for position, candidate in enumerate(candidates)
            if Position(position) in selected
        ]
