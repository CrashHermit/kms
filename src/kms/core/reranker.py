"""OpenRouter reranking client for search result re-scoring."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import httpx

from kms import config


def _api_key() -> str:
    """Returns the configured reranker API key.

    Raises:
        RuntimeError: If neither the reranker key nor the OpenRouter
            key is configured.
    """
    settings = config.get_settings()
    key = settings.reranker.api_key or settings.models.openrouter_api_key
    if not key:
        raise RuntimeError(
            'KMS_RERANKER__API_KEY is not set (and no '
            'KMS_MODELS__OPENROUTER_API_KEY to fall back to). Export '
            'your API key before calling the reranker.'
        )
    return key


def is_configured() -> bool:
    """True if a reranker API key is configured."""
    settings = config.get_settings()
    return bool(settings.reranker.api_key or settings.models.openrouter_api_key)


class Reranker:
    """Reranks document candidates against a query via an HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        settings = config.get_settings().reranker
        self.base_url = (base_url or settings.base_url).rstrip('/')
        self.model = model or settings.model
        self.api_key = api_key
        self.timeout = (
            timeout if timeout is not None else settings.timeout_seconds
        )
        self._client: httpx.AsyncClient | None = None

    async def _client_for(self) -> httpx.AsyncClient:
        """Returns the lazily-created HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={'Authorization': f'Bearer {self._require_key()}'},
            )
        return self._client

    def _require_key(self) -> str:
        """Returns the constructor key or the configured key."""
        return self.api_key or _api_key()

    async def rerank(
        self,
        query: str | dict[str, Any],
        documents: Sequence[str | dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Reranks the documents and returns the scored results.

        Args:
            query: The query, as text or multimodal content.
            documents: The candidate documents, as text or multimodal
                content.
            top_n: Maximum number of reranked results to return.

        Returns:
            The reranker's ``results`` list, highest score first.

        Raises:
            RuntimeError: If the rerank request fails.
        """
        client = await self._client_for()
        payload: dict[str, Any] = {
            'model': self.model,
            'query': query,
            'documents': list(documents),
        }
        if top_n is not None:
            payload['top_n'] = top_n

        response = await client.post(f'{self.base_url}/rerank', json=payload)
        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f'rerank request failed with HTTP '
                f'{response.status_code}: {body}'
            )
        return response.json().get('results', [])

    async def aclose(self) -> None:
        """Closes the HTTP client if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def reranker() -> Reranker:
    """Returns the shared Reranker, configured from the settings."""
    return Reranker()
