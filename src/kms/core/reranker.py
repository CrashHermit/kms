"""
Central reranker configuration for post-retrieval relevance scoring.

The counterpart to ``core.embeddings`` for ranking: a thin async httpx
client over an OpenAI-compatible ``/rerank`` endpoint, defaulting to
OpenRouter so any reranker OpenRouter serves works — Cohere
(``cohere/rerank-v3.5``), NVIDIA multimodal (``nvidia/llama-nemotron-
rerank-vl-1b-v2``), and the rest are just ``RERANK_MODEL`` changes.

Two document shapes, chosen by the caller:

* **Text-only:** pass a flat list of strings.
* **Multimodal:** pass a list of dicts, each optionally carrying
  ``"text"`` and/or ``"image"`` (URL or base64 data URI).  At least one
  of the two is required per document.

Config, all env-driven:

* ``RERANK_MODEL`` — the model id, default ``cohere/rerank-v3.5``.
* ``RERANK_BASE_URL`` — the endpoint root, default OpenRouter's
  ``https://openrouter.ai/api/v1``.
* ``RERANK_API_KEY`` — the key; falls back to ``OPENROUTER_API_KEY``.
"""

import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

RERANK_MODEL_ENV = 'RERANK_MODEL'
RERANK_BASE_URL_ENV = 'RERANK_BASE_URL'
RERANK_API_KEY_ENV = 'RERANK_API_KEY'

DEFAULT_RERANK_MODEL = 'cohere/rerank-v3.5'
DEFAULT_RERANK_BASE_URL = 'https://openrouter.ai/api/v1'

TIMEOUT_SECONDS = 60.0


def _api_key() -> str:
    """The reranker API key, falling back to the OpenRouter key."""
    key = os.environ.get(RERANK_API_KEY_ENV) or os.environ.get(
        'OPENROUTER_API_KEY'
    )
    if not key:
        raise RuntimeError(
            f'{RERANK_API_KEY_ENV} is not set (and no OPENROUTER_API_KEY '
            f'to fall back to). Export your API key before calling the '
            f'reranker.'
        )
    return key


def is_configured() -> bool:
    """Whether a reranker target is configured.

    The base URL defaults to OpenRouter, so only the key matters.
    """
    return bool(
        os.environ.get(RERANK_API_KEY_ENV)
        or os.environ.get('OPENROUTER_API_KEY')
    )


class Reranker:
    """A thin async client for an OpenAI-compatible rerank endpoint.

    Args:
        base_url: The endpoint root, e.g. OpenRouter's
            ``https://openrouter.ai/api/v1``.
        model: The reranker model id, e.g. ``cohere/rerank-v3.5``.
        api_key: The bearer token.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_RERANK_BASE_URL,
        model: str = DEFAULT_RERANK_MODEL,
        api_key: str | None = None,
        *,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _client_for(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={'Authorization': f'Bearer {self._require_key()}'},
            )
        return self._client

    def _require_key(self) -> str:
        return self.api_key or _api_key()

    async def rerank(
        self,
        query: str | dict[str, Any],
        documents: Sequence[str | dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score every document against *query*.

        Args:
            query: The search query — a string for text-only, or a
                dict with ``text`` and/or ``image`` keys.
            documents: The candidates to score.  Strings are treated as
                text-only documents; dicts may carry ``text`` and/or
                ``image`` (URL or base64 data URI).
            top_n: If set, return only the top *n* results.

        Returns:
            One dict per result:
            ``{index, relevance_score, document: {text, image?}}``,
            sorted by relevance descending.
        """
        client = await self._client_for()
        payload: dict[str, Any] = {
            'model': self.model,
            'query': query,
            'documents': list(documents),
        }
        if top_n is not None:
            payload['top_n'] = top_n

        response = await client.post(
            f'{self.base_url}/rerank', json=payload
        )
        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f'rerank request failed with HTTP '
                f'{response.status_code}: {body}'
            )
        return response.json().get('results', [])

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def reranker() -> Reranker:
    """The shared reranker, created once and reused.

    Built from ``RERANK_MODEL``, ``RERANK_BASE_URL``, and the API key
    (``RERANK_API_KEY`` with an ``OPENROUTER_API_KEY`` fallback).
    """
    return Reranker(
        base_url=os.environ.get(
            RERANK_BASE_URL_ENV, DEFAULT_RERANK_BASE_URL
        ),
        model=os.environ.get(RERANK_MODEL_ENV, DEFAULT_RERANK_MODEL),
    )
