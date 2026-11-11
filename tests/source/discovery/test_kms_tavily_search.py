"""Behavioral tests for Tavily search normalization."""

import json

import httpx
import pytest
from tavily import AsyncTavilyClient
from tavily.errors import InvalidAPIKeyError

from kms.source.discovery.search import TavilyWebClient
from kms.source.discovery.settings import TavilySettings


def _client(
    transport: httpx.AsyncBaseTransport,
) -> tuple[AsyncTavilyClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(
        transport=transport,
        base_url='https://api.tavily.com',
    )
    return (
        AsyncTavilyClient(
            api_key='tavily-secret',
            api_base_url='https://api.tavily.com',
            client=http_client,
        ),
        http_client,
    )


@pytest.mark.anyio
async def test_search_sends_configured_discovery_options_and_normalizes_candidates() -> (
    None
):
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'answer': 'not a candidate',
                'results': [
                    {
                        'title': 'Source title',
                        'url': 'https://public.example/source',
                        'content': 'Discovery snippet',
                        'score': 0.73,
                    }
                ],
            },
        )

    sdk, http_client = _client(httpx.MockTransport(respond))
    settings = TavilySettings(
        api_key='tavily-secret', search_depth='advanced', result_limit=7
    )
    try:
        results = await TavilyWebClient(settings, sdk).search('question')
    finally:
        await http_client.aclose()

    assert len(results) == 1
    assert results[0].model_dump() == {
        'title': 'Source title',
        'url': 'https://public.example/source',
        'snippet': 'Discovery snippet',
    }
    assert requests == [
        {
            'query': 'question',
            'search_depth': 'advanced',
            'include_answer': False,
            'include_raw_content': False,
            'max_results': 7,
            'auto_parameters': False,
        }
    ]


@pytest.mark.anyio
async def test_tavily_auth_failure_propagates() -> None:
    request_count = 0

    def unauthorized(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(401, json={'detail': {'error': 'invalid key'}})

    sdk, http_client = _client(httpx.MockTransport(unauthorized))
    try:
        with pytest.raises(InvalidAPIKeyError):
            await TavilyWebClient(
                TavilySettings(api_key='tavily-secret'), sdk
            ).search('question')
        assert request_count == 1
    finally:
        await http_client.aclose()
