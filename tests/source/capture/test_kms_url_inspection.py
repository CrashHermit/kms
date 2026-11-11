"""Behavioral and network-safety tests for public URL inspection."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from kms.source.capture.inspection import (
    ExternalRetrievalError,
    PublicURLInspector,
)

_PUBLIC_IP = '93.184.216.34'


async def _public_resolver(hostname: str, port: int) -> list[str]:
    assert port in {80, 443}
    return [_PUBLIC_IP]


def test_inspection_pins_each_redirect_and_preserves_provenance() -> None:
    async def exercise() -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.headers['host'] == 'first.example':
                return httpx.Response(
                    302, headers={'location': 'https://second.example/file'}
                )
            return httpx.Response(
                200,
                headers={'content-type': 'image/png'},
                stream=_NoReadStream(),
            )

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            headers={'Authorization': 'secret', 'Cookie': 'private=1'},
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        try:
            inspected = await PublicURLInspector(
                client, max_redirects=2, resolver=_public_resolver
            ).inspect('https://first.example/start')
        finally:
            await client.aclose()

        assert inspected.requested_url == 'https://first.example/start'
        assert inspected.final_url == 'https://second.example/file'
        assert inspected.media_type == 'image/png'
        assert [request.url.host for request in requests] == [
            _PUBLIC_IP,
            _PUBLIC_IP,
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

    asyncio.run(exercise())


class _NoReadStream(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        pytest.fail('declared media type must not read the body')
        yield b''

    async def aclose(self) -> None:
        self.closed = True


def test_declared_pdf_is_not_iterated_and_response_is_closed() -> None:
    async def exercise() -> None:
        stream = _NoReadStream()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={'content-type': 'application/pdf'},
                    stream=stream,
                )
            ),
            trust_env=False,
        ) as client:
            inspection = await PublicURLInspector(
                client, max_redirects=0, resolver=_public_resolver
            ).inspect('https://public.example/file')
        assert inspection.media_type == 'application/pdf'
        assert stream.closed

    asyncio.run(exercise())


def test_html_is_not_iterated_and_response_is_closed() -> None:
    async def exercise() -> None:
        stream = _NoReadStream()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={'content-type': 'text/html'}, stream=stream
                )
            ),
            trust_env=False,
        ) as client:
            inspection = await PublicURLInspector(
                client, max_redirects=0, resolver=_public_resolver
            ).inspect('https://public.example/page')
        assert inspection.media_type == 'text/html'
        assert stream.closed

    asyncio.run(exercise())


def test_ambiguous_pdf_reads_signature_only_and_closes_response() -> None:
    async def exercise() -> None:
        class PrefixStream(httpx.AsyncByteStream):
            closed = False
            chunks_requested = 0

            async def __aiter__(self) -> AsyncIterator[bytes]:
                self.chunks_requested += 1
                yield b'%PDF-'
                self.chunks_requested += 1
                raise AssertionError('inspection read past the PDF signature')

            async def aclose(self) -> None:
                self.closed = True

        stream = PrefixStream()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={'content-type': 'application/octet-stream'},
                    stream=stream,
                )
            ),
            trust_env=False,
        ) as client:
            inspection = await PublicURLInspector(
                client, max_redirects=0, resolver=_public_resolver
            ).inspect('https://public.example/download')
        assert inspection.media_type == 'application/pdf'
        assert stream.chunks_requested == 1
        assert stream.closed

    asyncio.run(exercise())


@pytest.mark.parametrize(
    'url',
    [
        'file:///etc/passwd',
        'data:image/png;base64,AAAA',
        'https://user:password@public.example/file',
        'https://public.example:8443/file',
    ],
)
def test_unsafe_url_forms_are_rejected(url: str) -> None:
    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: pytest.fail('unsafe URL connected')
            )
        ) as client:
            with pytest.raises(ExternalRetrievalError):
                await PublicURLInspector(
                    client, max_redirects=0, resolver=_public_resolver
                ).inspect(url)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    'url,addresses',
    [
        ('http://localhost/page', ['127.0.0.1']),
        ('http://private.example/page', ['10.1.2.3']),
        ('http://mixed.example/page', [_PUBLIC_IP, '192.168.1.2']),
    ],
)
def test_nonpublic_destinations_are_rejected(
    url: str, addresses: list[str]
) -> None:
    async def exercise() -> None:
        async def resolver(hostname: str, port: int) -> list[str]:
            return addresses

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: pytest.fail('private URL connected')
            )
        ) as client:
            with pytest.raises(ExternalRetrievalError):
                await PublicURLInspector(
                    client, max_redirects=0, resolver=resolver
                ).inspect(url)

    asyncio.run(exercise())


def test_private_redirect_is_blocked_before_second_request() -> None:
    async def exercise() -> None:
        requests: list[httpx.Request] = []

        async def resolver(hostname: str, port: int) -> list[str]:
            return (
                ['127.0.0.1'] if hostname == 'private.example' else [_PUBLIC_IP]
            )

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                302, headers={'location': 'https://private.example/file'}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as client:
            with pytest.raises(ExternalRetrievalError):
                await PublicURLInspector(
                    client, max_redirects=2, resolver=resolver
                ).inspect('https://public.example/file')
        assert len(requests) == 1

    asyncio.run(exercise())


def test_unsupported_media_and_compressed_ambiguous_media_are_rejected() -> (
    None
):
    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={'content-type': 'application/zip'},
                    content=b'zip',
                )
            )
        ) as client:
            with pytest.raises(ExternalRetrievalError, match='Unsupported'):
                await PublicURLInspector(
                    client, max_redirects=0, resolver=_public_resolver
                ).inspect('https://public.example/file')

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={
                        'content-type': 'application/octet-stream',
                        'content-encoding': 'gzip',
                    },
                    stream=_NoReadStream(),
                )
            )
        ) as client:
            with pytest.raises(ExternalRetrievalError, match='Compressed'):
                await PublicURLInspector(
                    client, max_redirects=0, resolver=_public_resolver
                ).inspect('https://public.example/file')

    asyncio.run(exercise())
