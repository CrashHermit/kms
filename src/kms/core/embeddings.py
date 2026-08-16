"""Voyage multimodal embeddings: configuration, batching, ranking."""

import math
from collections.abc import Sequence
from functools import lru_cache

import httpx

from kms import config
from kms.core import content


def is_configured() -> bool:
    """True if a Voyage API key is configured."""
    return bool(config.get_settings().embeddings.api_key)


class Embedder:
    """Embeds text and image content via the Voyage multimodal API.

    Every input — text, image, or a mix — is sent as multimodal
    content parts, so there is a single embedding path.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
        batch_size: int | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = config.get_settings().embeddings
        self.model = model or settings.model
        self.api_key = api_key
        self.timeout = (
            timeout if timeout is not None else settings.timeout_seconds
        )
        self.batch_size = (
            batch_size if batch_size is not None else settings.batch_size
        )
        self.dimension = (
            dimension if dimension is not None else settings.dimension
        )
        self.base_url = settings.base_url.rstrip('/')
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
        """Returns the API key from the constructor or the config.

        Raises:
            RuntimeError: If no key is configured.
        """
        if self.api_key:
            return self.api_key
        key = config.get_settings().embeddings.api_key
        if not key:
            raise RuntimeError(
                'KMS_EMBEDDINGS__API_KEY is not set. Export your Voyage '
                'API key before running a vector pass.'
            )
        return key

    async def _embed_batch(
        self, contents: Sequence[content.Content]
    ) -> list[list[float]]:
        """Embeds one batch of contents via the multimodal endpoint.

        Args:
            contents: The batch to embed.

        Returns:
            One embedding vector per content, in input order.

        Raises:
            RuntimeError: If the request fails or the response does not
                match the requested batch size.
        """
        client = await self._client_for()
        inputs = [{'content': item.embedding_blocks()} for item in contents]
        response = await client.post(
            f'{self.base_url}/multimodalembeddings',
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
        vectors = [item['embedding'] for item in data]
        invalid = [
            len(vector) for vector in vectors if len(vector) != self.dimension
        ]
        if invalid:
            raise RuntimeError(
                f'embedding dimension mismatch: configured {self.dimension}, '
                f'received {invalid[0]}'
            )
        return vectors

    async def embed(
        self, contents: Sequence[content.Content]
    ) -> list[list[float]]:
        """Embeds all contents, batching under the configured size.

        Args:
            contents: The contents to embed.

        Returns:
            One embedding vector per content, in input order.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(contents), self.batch_size):
            batch = contents[start : start + self.batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def aclose(self) -> None:
        """Closes the HTTP client if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    """Returns the shared Embedder, created lazily."""
    return Embedder()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Computes cosine similarity between two vectors.

    Args:
        left: The first vector.
        right: The second vector.

    Returns:
        The cosine similarity, or 0.0 when either vector is zero.

    Raises:
        ValueError: If the vectors differ in length.
    """
    if len(left) != len(right):
        raise ValueError(f'vector length mismatch: {len(left)} != {len(right)}')
    dot = sum(
        left_component * right_component
        for left_component, right_component in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def top_k(
    query: list[float],
    candidates: list[tuple[object, list[float]]],
    k: int | None = None,
    threshold: float | None = None,
) -> list[tuple[object, float]]:
    """Ranks candidates by cosine similarity to the query vector.

    Args:
        query: The query vector.
        candidates: ``(key, vector)`` pairs to rank.
        k: Maximum number of results to return.
        threshold: Optional minimum similarity; lower scores are dropped.

    Returns:
        The top k ``(key, score)`` pairs, highest score first.
    """
    if k is None:
        k = config.get_settings().stages.search.top_k
    scored = [
        (key, cosine_similarity(query, vector)) for key, vector in candidates
    ]
    if threshold is not None:
        scored = [entry for entry in scored if entry[1] >= threshold]
    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored[:k]
