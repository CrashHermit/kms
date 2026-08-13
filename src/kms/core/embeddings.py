import math
import os
from collections.abc import Sequence
from functools import lru_cache

import httpx

from kms.core import content

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

VOYAGE_API_KEY_ENV = 'VOYAGE_API_KEY'
EMBEDDING_MODEL_ENV = 'EMBEDDING_MODEL'
VOYAGE_BASE_URL = 'https://api.voyageai.com/v1'
DEFAULT_EMBEDDING_MODEL = 'voyage-multimodal-3.5'
BATCH_SIZE = 200

TIMEOUT_SECONDS = 60.0


def is_configured() -> bool:
    """True if a Voyage API key is configured."""
    return bool(os.environ.get(VOYAGE_API_KEY_ENV))


class Embedder:
    """Embeds text and image content via the Voyage multimodal API.

    Every input — text, image, or a mix — is sent as multimodal
    content parts, so there is a single embedding path.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
        *,
        timeout: float = TIMEOUT_SECONDS,
        batch_size: int = BATCH_SIZE,
    ) -> None:
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
        if self.api_key:
            return self.api_key
        key = os.environ.get(VOYAGE_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f'{VOYAGE_API_KEY_ENV} is not set. Export your Voyage API '
                f'key before running a vector pass.'
            )
        return key

    async def _embed_batch(
        self, contents: Sequence[content.Content]
    ) -> list[list[float]]:
        client = await self._client_for()
        inputs = [{'content': item.embedding_blocks()} for item in contents]
        response = await client.post(
            f'{VOYAGE_BASE_URL}/multimodalembeddings',
            json={'model': self.model, 'inputs': inputs},
        )
        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(
                f'embedding request failed with HTTP '
                f'{response.status_code}: {body}'
            )
        data = response.json().get('data')
        if not data or len(data) != len(contents):
            raise RuntimeError(
                f'embedding response mismatch: asked for {len(contents)} '
                f'vector(s), got {len(data or [])}'
            )
        return [item['embedding'] for item in data]

    async def embed(
        self, contents: Sequence[content.Content]
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(contents), self.batch_size):
            batch = contents[start : start + self.batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    """Returns the shared Embedder, created lazily."""
    return Embedder(
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
