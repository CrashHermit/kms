"""Provider-neutral multimodal reranking interfaces."""

from typing import Protocol, TypedDict

from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent


class RerankResult(TypedDict):
    """One reranker result in returned relevance order."""

    index: int
    relevance_score: float


class Reranker(Protocol):
    """Rerank multimodal documents for one query."""

    async def rerank(
        self,
        query: MultimodalContent,
        documents: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return original candidate indexes in relevance order."""
        ...
