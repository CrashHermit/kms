"""Behavioral tests for Tavily normalization and public-media boundaries."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from tavily import AsyncTavilyClient
from tavily.errors import InvalidAPIKeyError

from kms2.config.research import TavilySettings
from kms2.core.model.research.retrieval import WebExtractionBatch
from kms2.retrieval.download import ExternalRetrievalError, PublicMediaClient
from kms2.retrieval.tavily import TavilyWebClient


def _sdk(
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


def _settings() -> TavilySettings:
    return TavilySettings(api_key='tavily-secret')


def test_tavily_extract_batches_and_maps_by_url() -> None:
    asyncio.run(_tavily_extract_batches_and_maps_by_url())


async def _tavily_extract_batches_and_maps_by_url() -> None:
    urls = [f'https://public.example/page/{index}' for index in range(21)]
    calls: list[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        batch = payload['urls']
        calls.append(batch)
        failures = []
        results = []
        for url in reversed(batch):
            if url == urls[5]:
                failures.append({'url': url, 'error': 'unavailable'})
            else:
                results.append(
                    {
                        'url': url,
                        'raw_content': f'Markdown for {url}',
                        'images': [f'{url}/image.png'],
                    }
                )
        return httpx.Response(
            200,
            json={'results': results, 'failed_results': failures},
        )

    sdk, http_client = _sdk(httpx.MockTransport(respond))
    client = TavilyWebClient(
        _settings(),
        sdk,
        resolver=_public_resolver,
    )
    try:
        batch = await client.extract(urls + [urls[0]])
    finally:
        await http_client.aclose()

    assert [len(group) for group in calls] == [20, 1]
    assert calls[0] == urls[:20]
    assert calls[1] == urls[20:]
    assert [result.url for result in batch.results] == [
        url for url in urls if url != urls[5]
    ]
    assert [(failure.url, failure.error) for failure in batch.failures] == [
        (urls[5], 'unavailable')
    ]
    assert next(
        result for result in batch.results if result.url == urls[18]
    ).raw_markdown == (f'Markdown for {urls[18]}')


def test_tavily_search_uses_discovery_fields_without_generated_answer() -> None:
    asyncio.run(_tavily_search_uses_discovery_fields_without_generated_answer())


async def _tavily_search_uses_discovery_fields_without_generated_answer() -> (
    None
):
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                'answer': 'should not become a result',
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

    sdk, http_client = _sdk(httpx.MockTransport(respond))
    client = TavilyWebClient(_settings(), sdk)
    try:
        hits = await client.search('question')
    finally:
        await http_client.aclose()

    assert len(hits) == 1
    assert hits[0].title == 'Source title'
    assert hits[0].snippet == 'Discovery snippet'
    assert not hasattr(hits[0], 'answer')
    assert requests == [
        {
            'query': 'question',
            'search_depth': 'basic',
            'include_answer': False,
            'include_raw_content': False,
            'max_results': 5,
            'auto_parameters': False,
        }
    ]


def test_tavily_auth_failure_propagates_and_empty_extract_is_local() -> None:
    asyncio.run(_tavily_auth_failure_propagates_and_empty_extract_is_local())


async def _tavily_auth_failure_propagates_and_empty_extract_is_local() -> None:
    request_count = 0

    def unauthorized(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(401, json={'detail': {'error': 'invalid key'}})

    sdk, http_client = _sdk(httpx.MockTransport(unauthorized))
    client = TavilyWebClient(_settings(), sdk)
    try:
        assert await client.extract([]) == WebExtractionBatch()
        assert request_count == 0
        with pytest.raises(InvalidAPIKeyError):
            await client.search('question')
        assert request_count == 1
    finally:
        await http_client.aclose()


async def _public_resolver(hostname: str, port: int) -> list[str]:
    assert port in {80, 443}
    return ['93.184.216.34']


def test_downloader_pins_public_ip_and_strips_credentials_across_redirects() -> (
    None
):
    asyncio.run(
        _downloader_pins_public_ip_and_strips_credentials_across_redirects()
    )


async def _downloader_pins_public_ip_and_strips_credentials_across_redirects() -> (
    None
):
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers['host'] == 'first.example':
            return httpx.Response(
                302,
                headers={'location': 'https://second.example/image.png'},
            )
        return httpx.Response(
            200,
            headers={'content-type': 'image/png'},
            content=b'png-bytes',
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        headers={
            'Authorization': 'Bearer tavily-secret',
            'Cookie': 'session=private',
        },
        cookies={'jar': 'must-not-leak'},
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    client = PublicMediaClient(
        http_client,
        max_download_bytes=1024,
        max_redirects=2,
        resolver=_public_resolver,
    )
    try:
        media = await client.download('https://first.example/start')
    finally:
        await http_client.aclose()

    assert media.requested_url == 'https://first.example/start'
    assert media.final_url == 'https://second.example/image.png'
    assert media.data == b'png-bytes'
    assert [request.url.host for request in requests] == [
        '93.184.216.34',
        '93.184.216.34',
    ]
    assert [request.headers['host'] for request in requests] == [
        'first.example',
        'second.example',
    ]
    assert [request.extensions['sni_hostname'] for request in requests] == [
        'first.example',
        'second.example',
    ]
    for request in requests:
        assert 'authorization' not in request.headers
        assert 'cookie' not in request.headers


@pytest.mark.parametrize(
    'url,resolved',
    [
        ('http://localhost/image.png', ['127.0.0.1']),
        ('http://private.example/image.png', ['10.1.2.3']),
        ('http://linklocal.example/image.png', ['169.254.10.2']),
        (
            'http://mixed.example/image.png',
            ['93.184.216.34', '192.168.1.2'],
        ),
        ('http://[::ffff:127.0.0.1]/image.png', []),
    ],
)
def test_downloader_rejects_nonpublic_destinations(
    url: str,
    resolved: list[str],
) -> None:
    asyncio.run(_downloader_rejects_nonpublic_destinations(url, resolved))


async def _downloader_rejects_nonpublic_destinations(
    url: str,
    resolved: list[str],
) -> None:
    async def resolver(hostname: str, port: int) -> list[str]:
        return resolved

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: pytest.fail('blocked URL reached transport')
        )
    )
    client = PublicMediaClient(
        http_client,
        max_download_bytes=1024,
        max_redirects=2,
        resolver=resolver,
    )
    try:
        with pytest.raises(ExternalRetrievalError):
            await client.download(url)
    finally:
        await http_client.aclose()


@pytest.mark.parametrize(
    'url',
    [
        'file:///etc/passwd',
        'data:image/png;base64,AAAA',
        'https://user:password@public.example/image.png',
        'https://public.example:8443/image.png',
    ],
)
def test_downloader_rejects_unsafe_url_forms(url: str) -> None:
    asyncio.run(_downloader_rejects_unsafe_url_forms(url))


async def _downloader_rejects_unsafe_url_forms(url: str) -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: pytest.fail('invalid URL reached transport')
        )
    )
    client = PublicMediaClient(
        http_client,
        max_download_bytes=1024,
        max_redirects=2,
        resolver=_public_resolver,
    )
    try:
        with pytest.raises(ExternalRetrievalError):
            await client.download(url)
    finally:
        await http_client.aclose()


def test_downloader_revalidates_redirect_before_request() -> None:
    asyncio.run(_downloader_revalidates_redirect_before_request())


async def _downloader_revalidates_redirect_before_request() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            302,
            headers={'location': 'http://private.example/secret'},
        )

    async def resolver(hostname: str, port: int) -> list[str]:
        if hostname == 'private.example':
            return ['127.0.0.1']
        return ['93.184.216.34']

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = PublicMediaClient(
        http_client,
        max_download_bytes=1024,
        max_redirects=3,
        resolver=resolver,
    )
    try:
        with pytest.raises(ExternalRetrievalError):
            await client.download('https://first.example/image.png')
    finally:
        await http_client.aclose()
    assert len(requests) == 1


def test_downloader_counts_streamed_bytes_without_content_length() -> None:
    asyncio.run(_downloader_counts_streamed_bytes_without_content_length())


async def _downloader_counts_streamed_bytes_without_content_length() -> None:
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'0123'
            yield b'456789'

        async def aclose(self) -> None:
            return None

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={'content-type': 'application/pdf'},
                stream=Chunks(),
            )
        )
    )
    client = PublicMediaClient(
        http_client,
        max_download_bytes=4,
        max_redirects=0,
        resolver=_public_resolver,
    )
    try:
        with pytest.raises(ExternalRetrievalError, match='download size limit'):
            await client.download('https://public.example/book.pdf')
    finally:
        await http_client.aclose()


def test_downloader_rejects_unsupported_response_media() -> None:
    asyncio.run(_downloader_rejects_unsupported_response_media())


async def _downloader_rejects_unsupported_response_media() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={'content-type': 'text/html'},
                content=b'<html>not media</html>',
            )
        )
    )
    client = PublicMediaClient(
        http_client,
        max_download_bytes=1024,
        max_redirects=0,
        resolver=_public_resolver,
    )
    try:
        with pytest.raises(ExternalRetrievalError, match='Unsupported'):
            await client.download('https://public.example/image.png')
    finally:
        await http_client.aclose()
