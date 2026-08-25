"""Local HTTP embeddings and vector ranking helpers."""

import math
from collections.abc import Sequence
from functools import lru_cache

import httpx

from kms import config
from kms.core import models, serve


def is_configured() -> bool:
    return bool(config.get_settings().embeddings.model)


@lru_cache(maxsize=1)
def _http_client() -> httpx.AsyncClient:
    settings = config.get_settings().embeddings
    return httpx.AsyncClient(
        base_url=settings.base_url.rstrip('/'), timeout=settings.timeout_seconds
    )


class Embedder:
    """Lazily-loaded local HTTP embedding client."""

    def __init__(
        self,
        model: str | None = None,
        *,
        batch_size: int | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = config.get_settings().embeddings
        self.model = model or settings.model
        self.batch_size = batch_size or settings.batch_size
        self.dimension = dimension or settings.dimension

    @staticmethod
    def _validate(vectors: list[list[float]], dimension: int) -> list[list[float]]:
        invalid = [len(vector) for vector in vectors if len(vector) != dimension]
        if invalid:
            raise RuntimeError(
                f'embedding dimension mismatch: configured {dimension}, '
                f'received {invalid[0]}'
            )
        return vectors

    async def _request(self, values: list[str]) -> list[list[float]]:
        await serve.retrieval_server_manager().aensure_embedding_started()
        endpoint = config.get_settings().embeddings.base_url
        try:
            response = await _http_client().post(
                '/embeddings', json={'model': self.model, 'input': values}
            )
            response.raise_for_status()
            payload = response.json()
            data = payload['data']
            if not isinstance(data, list) or len(data) != len(values):
                raise ValueError('response data count mismatch')
            vectors: list[list[float] | None] = [None] * len(values)
            for item in data:
                index = item['index']
                vector = item['embedding']
                if (
                    not isinstance(index, int)
                    or index < 0
                    or index >= len(values)
                    or vectors[index] is not None
                    or not isinstance(vector, list)
                ):
                    raise ValueError('invalid or duplicated response index')
                vectors[index] = vector
            if any(vector is None for vector in vectors):
                raise ValueError('missing response index')
            return self._validate(vectors, self.dimension)  # type: ignore[arg-type]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f'local embedding endpoint {endpoint} failed: {exc}'
            ) from exc

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError('embed() accepts strings only')
        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            vectors.extend(await self._request(values[start : start + self.batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError('embed_query() accepts a string only')
        vectors = await self._request([text])
        if len(vectors) != 1:
            raise RuntimeError(f'query embedding returned {len(vectors)} vectors')
        return vectors[0]

    async def aclose(self) -> None:
        if _http_client.cache_info().currsize:
            await _http_client().aclose()


async def close_retrieval_clients() -> None:
    if _http_client.cache_info().currsize:
        await _http_client().aclose()
    _http_client.cache_clear()


async def embed_source_nodes(
    nodes: Sequence[models.SourceNode],
) -> list[list[float] | None]:
    texts: list[str] = []
    positions: list[int] = []
    for position, node in enumerate(nodes):
        if node.content and node.content.strip():
            texts.append(node.content)
            positions.append(position)
    if not texts:
        return [None] * len(nodes)
    vectors = await embedder().embed(texts)
    result: list[list[float] | None] = [None] * len(nodes)
    for position, vector in zip(positions, vectors, strict=True):
        result[position] = vector
    return result


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f'vector length mismatch: {len(left)} != {len(right)}')
    dot = sum(a * b for a, b in zip(left, right, strict=True))
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
    if k is None:
        k = config.get_settings().stages.search.top_k
    scored = [(key, cosine_similarity(query, vector)) for key, vector in candidates]
    if threshold is not None:
        scored = [entry for entry in scored if entry[1] >= threshold]
    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored[:k]
