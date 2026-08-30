"""Local HTTP text reranking client."""

import logging
import time
from collections.abc import Sequence
from functools import lru_cache
from numbers import Real
from typing import Any

import httpx

from kms import config
from kms.core import logs, serve

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    return bool(config.get_settings().reranker.model)


@lru_cache(maxsize=1)
def _http_client() -> httpx.AsyncClient:
    settings = config.get_settings().reranker
    return httpx.AsyncClient(
        base_url=settings.base_url.rstrip('/'), timeout=settings.timeout_seconds
    )


class Reranker:
    """Lazily-loaded local HTTP reranking client."""

    def __init__(self, model: str | None = None, *, batch_size: int | None = None) -> None:
        settings = config.get_settings().reranker
        self.model = model or settings.model
        self.batch_size = batch_size or settings.batch_size

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or any(
            not isinstance(document, str) for document in documents
        ):
            raise TypeError('rerank() accepts a text query and text documents')
        values = list(documents)
        if not values:
            return []
        started = time.perf_counter()
        await serve.retrieval_server_manager().aensure_reranker_started()
        body: dict[str, Any] = {
            'model': self.model,
            'query': query,
            'documents': values,
        }
        if top_n is not None:
            body['top_n'] = top_n
        endpoint = config.get_settings().reranker.base_url
        try:
            response = await _http_client().post('/rerank', json=body)
            response.raise_for_status()
            results = response.json()['results']
            if not isinstance(results, list) or not results:
                raise ValueError('invalid reranker result count')
            if len(results) > len(values):
                raise ValueError('invalid reranker result count')
            seen: set[int] = set()
            validated: list[dict[str, Any]] = []
            for result in results:
                index = result['index']
                score = result['relevance_score']
                if (
                    not isinstance(index, int)
                    or index < 0
                    or index >= len(values)
                    or index in seen
                    or not isinstance(score, Real)
                    or isinstance(score, bool)
                ):
                    raise ValueError('invalid or duplicated reranker result')
                seen.add(index)
                validated.append({'index': index, 'relevance_score': float(score)})
            validated.sort(
                key=lambda result: (-result['relevance_score'], result['index'])
            )
            selected = validated if top_n is None else validated[:top_n]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f'local reranker endpoint {endpoint} failed: {exc}') from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            'reranker: %d candidates -> %d selected; query=%s; '
            'documents=%s; results=%s; duration=%s ms',
            len(values),
            len(selected),
            logs.elide(query),
            logs.elide([logs.elide(doc) for doc in values]),
            logs.elide(selected),
            elapsed_ms,
        )
        return selected

    async def aclose(self) -> None:
        if _http_client.cache_info().currsize:
            await _http_client().aclose()


async def close_retrieval_clients() -> None:
    if _http_client.cache_info().currsize:
        await _http_client().aclose()
    _http_client.cache_clear()


@lru_cache(maxsize=1)
def reranker() -> Reranker:
    return Reranker()
