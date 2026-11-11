"""Behavioral tests for source search and selection."""

import pytest

from kms.source.discovery.discovery import SourceDiscovery
from kms.source.discovery.models import SourceCandidate
from kms.source.discovery.selection import (
    DiscoverySelection,
    DiscoverySelectionInput,
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _Web:
    def __init__(self, hits: list[SourceCandidate]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    async def search(self, query: str) -> list[SourceCandidate]:
        self.queries.append(query)
        return self.hits


class _Selector:
    def __init__(self, positions: list[int]) -> None:
        self.positions = positions
        self.requests: list[DiscoverySelectionInput] = []

    async def acall(
        self,
        *,
        request: DiscoverySelectionInput,
    ) -> DiscoverySelection:
        self.requests.append(request)
        return DiscoverySelection(result_positions=self.positions)


def _candidate(title: str | None, url: str, snippet: str) -> SourceCandidate:
    return SourceCandidate(title=title, url=url, snippet=snippet)


@pytest.mark.anyio
async def test_searches_exact_question_deduplicates_urls_and_restores_search_order() -> (
    None
):
    first = _candidate('First', 'https://example.test/a', 'first snippet')
    duplicate = _candidate('Changed', first.url, 'changed duplicate')
    second = _candidate(None, 'https://example.test/b', 'second snippet')
    third = _candidate('Third', 'https://example.test/c', 'third snippet')
    web = _Web([first, duplicate, second, third])
    selector = _Selector([2, 0])

    results = await SourceDiscovery(web, selector).discover(
        '  exact question?  '
    )

    assert web.queries == ['  exact question?  ']
    assert len(selector.requests) == 1
    request = selector.requests[0]
    assert request.question == '  exact question?  '
    assert [
        (result.position, result.title, result.url, result.snippet)
        for result in request.results
    ] == [
        (0, 'First', first.url, 'first snippet'),
        (1, None, second.url, 'second snippet'),
        (2, 'Third', third.url, 'third snippet'),
    ]
    assert results == [first, third]
    assert all(not hasattr(result, 'position') for result in results)


@pytest.mark.anyio
async def test_no_hits_skip_selection_and_empty_selection_returns_no_candidates() -> (
    None
):
    empty_selector = _Selector([0])
    empty_web = _Web([])
    assert (
        await SourceDiscovery(empty_web, empty_selector).discover('question')
        == []
    )
    assert empty_web.queries == ['question']
    assert empty_selector.requests == []

    web = _Web([_candidate('Title', 'https://example.test/a', 'snippet')])
    selector = _Selector([])
    assert await SourceDiscovery(web, selector).discover('question') == []
    assert len(selector.requests) == 1
