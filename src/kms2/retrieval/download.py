"""SSRF-resistant downloading of public image and PDF media."""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import httpx

from kms2.core.model.research.retrieval import DownloadedMedia

_DnsResolver = Callable[[str, int], Awaitable[list[str]]]
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class ExternalRetrievalError(ValueError):
    """An external resource is blocked, unsupported, or exceeds its limits."""


class PublicMediaClient:
    """Download bounded public image or PDF bytes without following unsafe hops."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        max_download_bytes: int,
        max_redirects: int,
        resolver: _DnsResolver | None = None,
    ) -> None:
        self._client = client
        self._max_download_bytes = max_download_bytes
        self._max_redirects = max_redirects
        self._resolver = resolver

    async def download(self, url: str) -> DownloadedMedia:
        """Fetch a public image or PDF, pinning every request to resolved DNS."""
        requested_url = url
        current_url = url
        redirects = 0
        while True:
            parsed, addresses = await _resolve_public_url(
                current_url, resolver=self._resolver
            )
            request_url = _pinned_url(parsed, addresses[0])
            headers = {
                'Accept': 'image/*, application/pdf',
                'User-Agent': 'KMS2 external research fetcher',
                'Host': _host_header(parsed),
            }
            request = self._client.build_request(
                'GET', request_url, headers=headers
            )
            request.headers.clear()
            request.headers.update(headers)
            if parsed.scheme == 'https':
                request.extensions['sni_hostname'] = parsed.hostname

            response = await self._client.send(
                request,
                stream=True,
                follow_redirects=False,
            )
            try:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get('location')
                    if not location:
                        response.raise_for_status()
                    if redirects >= self._max_redirects:
                        raise ExternalRetrievalError(
                            'External media exceeded the redirect limit'
                        )
                    current_url = urljoin(current_url, location)
                    redirects += 1
                    continue

                response.raise_for_status()
                media_type = response.headers.get('content-type', '')
                media_type = media_type.split(';', 1)[0].strip().lower()
                if not (
                    media_type.startswith('image/')
                    or media_type == 'application/pdf'
                ):
                    raise ExternalRetrievalError(
                        f'Unsupported external media type: {media_type or "missing"}'
                    )

                chunks: list[bytes] = []
                byte_count = 0
                async for chunk in response.aiter_bytes():
                    byte_count += len(chunk)
                    if byte_count > self._max_download_bytes:
                        raise ExternalRetrievalError(
                            'External media exceeded the download size limit'
                        )
                    chunks.append(chunk)
                data = b''.join(chunks)
                if not data:
                    raise ExternalRetrievalError(
                        'External media response is empty'
                    )
                return DownloadedMedia(
                    requested_url=requested_url,
                    final_url=current_url,
                    retrieved_at=datetime.now(UTC),
                    media_type=media_type,
                    data=data,
                )
            finally:
                await response.aclose()


async def validate_public_url(
    url: str,
    *,
    resolver: _DnsResolver | None = None,
) -> None:
    """Require an HTTP(S) URL whose complete DNS answer set is globally routable."""
    await _resolve_public_url(url, resolver=resolver)


async def _resolve_public_url(
    url: str,
    *,
    resolver: _DnsResolver | None,
) -> tuple[SplitResult, list[str]]:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in {'http', 'https'}:
            raise ExternalRetrievalError('Only HTTP(S) URLs are allowed')
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ExternalRetrievalError(
                'External URLs must have a hostname and no embedded credentials'
            )
        port = parsed.port
        expected_port = 443 if scheme == 'https' else 80
        if port is not None and port != expected_port:
            raise ExternalRetrievalError(
                'Nondefault external URL ports are blocked'
            )
        effective_port = port or expected_port
    except ValueError as error:
        if isinstance(error, ExternalRetrievalError):
            raise
        raise ExternalRetrievalError('Malformed external URL') from error

    hostname = parsed.hostname
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        dns_resolver = resolver or _resolve_dns
        try:
            addresses = await dns_resolver(hostname, effective_port)
        except socket.gaierror as error:
            raise ExternalRetrievalError(
                'External hostname could not be resolved'
            ) from error
    else:
        addresses = [str(address)]

    if not addresses:
        raise ExternalRetrievalError('External hostname has no addresses')
    for item in addresses:
        try:
            candidate = ipaddress.ip_address(item)
        except ValueError as error:
            raise ExternalRetrievalError(
                'External hostname returned an invalid address'
            ) from error
        mapped = (
            candidate.ipv4_mapped
            if isinstance(candidate, ipaddress.IPv6Address)
            else None
        )
        if not candidate.is_global or (
            mapped is not None and not mapped.is_global
        ):
            raise ExternalRetrievalError(
                'External hostname resolves to a non-public address'
            )
    return parsed, addresses


async def _resolve_dns(hostname: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        hostname,
        port,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return list(dict.fromkeys(record[4][0] for record in records))


def _host_header(parsed: SplitResult) -> str:
    hostname = parsed.hostname
    if ':' in hostname and not hostname.startswith('['):
        hostname = f'[{hostname}]'
    if parsed.port is not None:
        return f'{hostname}:{parsed.port}'
    return hostname


def _pinned_url(parsed: SplitResult, address: str) -> str:
    if ':' in address:
        address = f'[{address}]'
    if parsed.port is not None:
        address = f'{address}:{parsed.port}'
    return urlunsplit(
        (parsed.scheme, address, parsed.path or '/', parsed.query, '')
    )
