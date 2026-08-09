import math
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

EMBEDDING_MODEL_ENV = 'EMBEDDING_MODEL'
EMBEDDING_BASE_URL_ENV = 'EMBEDDING_BASE_URL'
EMBEDDING_API_KEY_ENV = 'EMBEDDING_API_KEY'

DEFAULT_EMBEDDING_MODEL = 'voyage-multimodal-3.5'
DEFAULT_EMBEDDING_BASE_URL = 'https://openrouter.ai/api/v1'
BATCH_SIZE = 200

TIMEOUT_SECONDS = 60.0


def _api_key() -> str:
    key = os.environ.get(EMBEDDING_API_KEY_ENV) or os.environ.get(
        'OPENROUTER_API_KEY'
    )
    if not key:
        raise RuntimeError(
            f'{EMBEDDING_API_KEY_ENV} is not set (and no OPENROUTER_API_KEY '
            f'to fall back to). Export your API key before running a vector '
            f'pass.'
        )
    return key


def is_configured() -> bool:
    return bool(
        os.environ.get(EMBEDDING_API_KEY_ENV)
        or os.environ.get('OPENROUTER_API_KEY')
    )


class Embedder:
    def __init__(
        self,
        base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
        *,
        timeout: float = TIMEOUT_SECONDS,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
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

    async def _embed_batch(
        self, texts: Sequence[str | dict[str, Any]]
    ) -> list[list[float]]:
        client = await self._client_for()
        response = await client.post(
            f'{self.base_url}/embeddings',
            json={'model': self.model, 'input': list(texts)},
        )
        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f'embedding request failed with HTTP '
                f'{response.status_code}: {body}'
            )
        data = response.json().get('data')
        if not data or len(data) != len(texts):
            raise RuntimeError(
                f'embedding response mismatch: asked for {len(texts)} '
                f'vector(s), got {len(data or [])}'
            )
        return [item['embedding'] for item in data]

    async def embed(
        self, texts: Sequence[str | dict[str, Any]]
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder(
        base_url=os.environ.get(
            EMBEDDING_BASE_URL_ENV, DEFAULT_EMBEDDING_BASE_URL
        ),
        model=os.environ.get(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL),
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f'vector length mismatch: {len(left)} != {len(right)}')
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def top_k(
    query: list[float],
    candidates: list[tuple[object, list[float]]],
    k: int = 10,
    threshold: float | None = None,
) -> list[tuple[object, float]]:
    scored = [
        (key, cosine_similarity(query, vector)) for key, vector in candidates
    ]
    if threshold is not None:
        scored = [entry for entry in scored if entry[1] >= threshold]
    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored[:k]

