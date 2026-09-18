"""Provider-neutral reranking interfaces for KMS2."""

from typing import Protocol, TypedDict, runtime_checkable


class RerankResult(TypedDict):
    """One reranker result in returned relevance order."""

    index: int
    relevance_score: float


@runtime_checkable
class RerankerClient(Protocol):
    """Rerank documents for one query while preserving server results."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return relevance-ranked document indexes and scores."""
        ...


__all__ = ['RerankResult', 'RerankerClient']
