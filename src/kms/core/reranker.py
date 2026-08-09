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
    return bool(
        os.environ.get(RERANK_API_KEY_ENV)
        or os.environ.get('OPENROUTER_API_KEY')
    )


class Reranker:
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
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def reranker() -> Reranker:
    return Reranker(
        base_url=os.environ.get(RERANK_BASE_URL_ENV, DEFAULT_RERANK_BASE_URL),
        model=os.environ.get(RERANK_MODEL_ENV, DEFAULT_RERANK_MODEL),
    )

