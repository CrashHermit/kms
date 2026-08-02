"""
Central embedding configuration for the vector passes.

The counterpart to ``core.llm`` for vectors: the shared embedder and its
credentials live in one place, read from the environment, and every stage
that needs similarity injects the one shared instance. It lives in ``core``
because — unlike the graph tier's Neo4j driver — embeddings are a
cross-phase utility (concept discovery, fact dedup, relation matching) that
several phases touch.

One embedder: a thin async httpx client over an OpenAI-compatible
``/embeddings`` endpoint, defaulting to OpenRouter so any embedding model
OpenRouter serves works — Voyage AI (e.g.
``voyage-ai/voyage-multimodal-3.5``), OpenAI, BAAI/bge, and the rest are
just ``EMBEDDING_MODEL`` changes. Deliberately NOT ``dspy.Embedder``: the
passes need provider freedom (and later multimodal input) that a direct
endpoint call gives.

Config, all env-driven:

* ``EMBEDDING_MODEL`` — the model id, default ``voyage-multimodal-3.5``
  (Voyage's multimodal model, served by OpenRouter under its bare id —
  embedding models are not listed in OpenRouter's catalog but are accepted
  by the ``/embeddings`` endpoint).
* ``EMBEDDING_BASE_URL`` — the endpoint root, default OpenRouter's
  ``https://openrouter.ai/api/v1``.
* ``EMBEDDING_API_KEY`` — the key; falls back to the existing
  ``OPENROUTER_API_KEY``, so an OpenRouter-served model runs with zero new
  configuration.

The key is required on use, not import — the module stays importable
without credentials, and the pure similarity helpers
(``cosine_similarity``, ``top_k``) never need one.
"""

import math
import os
from collections.abc import Sequence
from functools import lru_cache

import httpx

# Load a local .env if present, guarded — same convenience as core.llm, and
# harmless when the module is imported before any core.llm import has
# already loaded it.
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

# How many texts one request may carry. Kept under the provider limits; the
# client splits larger batches across several requests.
BATCH_SIZE = 200

TIMEOUT_SECONDS = 60.0


def _api_key() -> str:
    """The embedding API key: ``EMBEDDING_API_KEY``, else the OpenRouter key.

    The OpenRouter fallback is deliberate — the default endpoint is
    OpenRouter and the repo already carries ``OPENROUTER_API_KEY``, so a
    Voyage/OpenAI/bge model runs with no new configuration.

    Returns:
        The key.

    Raises:
        RuntimeError: If neither env var is set.
    """
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
    """Whether an embedding target is configured.

    The base URL defaults to OpenRouter, so only the key matters. Lets the
    pipeline skip embedding gracefully when no key is wired — the persister
    then writes ``:Fact`` nodes without the embedding property.

    Returns:
        True if an embedding API key is available.
    """
    return bool(
        os.environ.get(EMBEDDING_API_KEY_ENV)
        or os.environ.get('OPENROUTER_API_KEY')
    )


class Embedder:
    """A thin async client for an OpenAI-compatible embeddings endpoint.

    Args:
        base_url: The endpoint root, e.g. OpenRouter's
            ``https://openrouter.ai/api/v1``.
        model: The embedding model id, e.g.
            ``voyage-ai/voyage-multimodal-3.5``.
        api_key: The bearer token.
        timeout: Per-request timeout in seconds.
        batch_size: Texts per request; larger batches are split.
    """

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
        """The shared per-embedder client, created lazily."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={'Authorization': f'Bearer {self._require_key()}'},
            )
        return self._client

    def _require_key(self) -> str:
        """The key, or a clear error on use."""
        return self.api_key or _api_key()

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """One request's vectors, in input order."""
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

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in input order.

        Texts are split into ``batch_size`` chunks; the vectors are returned
        concatenated so callers never see the batching.

        Args:
            texts: The strings to embed.

        Returns:
            One vector per input string, in the same order.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def aclose(self) -> None:
        """Close the underlying client, if one was opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    """The shared embedder, created once and reused.

    Built from ``EMBEDDING_MODEL``, ``EMBEDDING_BASE_URL``, and the API key
    (``EMBEDDING_API_KEY`` with an ``OPENROUTER_API_KEY`` fallback). Cached
    so every stage that calls this shares one instance.

    Returns:
        The shared embedder.
    """
    return Embedder(
        base_url=os.environ.get(
            EMBEDDING_BASE_URL_ENV, DEFAULT_EMBEDDING_BASE_URL
        ),
        model=os.environ.get(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL),
    )


# ============================================================================
# Pure similarity helpers (dependency-free, unit-testable without a model)
# ============================================================================


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 when either vector is all zeros (no direction to compare).

    Args:
        left: The first vector.
        right: The second vector.

    Returns:
        The cosine of the angle between them, in [-1, 1].

    Raises:
        ValueError: If the vectors differ in length.
    """
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
    """The k most similar candidates to a query vector.

    Args:
        query: The query vector.
        candidates: ``(key, vector)`` pairs; the key is opaque.
        k: How many to return.
        threshold: Optional minimum similarity; entries below it are dropped.

    Returns:
        The best entries as ``(key, similarity)``, best first.
    """
    scored = [
        (key, cosine_similarity(query, vector)) for key, vector in candidates
    ]
    if threshold is not None:
        scored = [entry for entry in scored if entry[1] >= threshold]
    scored.sort(key=lambda entry: entry[1], reverse=True)
    return scored[:k]
