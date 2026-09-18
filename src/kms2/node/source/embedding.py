"""LangGraph node for final source-block embeddings."""

from typing import Literal, TypedDict, cast

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import EmbeddingRequest, EmbeddingResult, SourcePage
from kms2.langgraph.source.state import SourceState


class EmbeddingWorkerState(TypedDict):
    """State supplied to the dispatched embedding worker."""

    embedding_request: EmbeddingRequest


class EmbeddingNode:
    """Dispatch, embed, and collect the final split source blocks."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def dispatch(
        self, state: SourceState
    ) -> list[Send] | Literal['embedding_collect']:
        """Dispatch the final split pages as one ordered embedding request."""
        if not any(page.blocks for page in state.split_pages):
            return 'embedding_collect'
        return [
            Send(
                'embedding_worker',
                {
                    'embedding_request': EmbeddingRequest(
                        pages=state.split_pages,
                    )
                },
            )
        ]

    async def worker(
        self, state: EmbeddingWorkerState
    ) -> dict[str, list[EmbeddingResult]]:
        """Embed the dispatched final blocks in page and block order."""
        pages = state['embedding_request'].pages
        blocks = [block for page in pages for block in page.blocks]
        embeddings = await self._client.embed(
            [cast(str, block.content) for block in blocks]
        )
        embedded_blocks = iter(embeddings)
        embedded_pages: list[SourcePage] = []
        for page in pages:
            page_blocks = [
                block.model_copy(update={'embedding': next(embedded_blocks)})
                for block in page.blocks
            ]
            embedded_pages.append(
                page.model_copy(update={'blocks': page_blocks})
            )
        return {'embedding_results': [EmbeddingResult(pages=embedded_pages)]}

    def collect(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Collect the one ordered embedding result for persistence."""
        return {
            'embedded_pages': (
                state.embedding_results[0].pages
                if state.embedding_results
                else state.split_pages
            )
        }
