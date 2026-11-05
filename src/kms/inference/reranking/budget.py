"""IONOS Qwen vision reranking pair limits."""

from dataclasses import dataclass

from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.core.tokenization.images import ImageMetadataCache
from kms.core.tokenization.text import TextTokenCounter

IONOS_MAX_PAIR_TOKENS = 32_768
IONOS_FRAMING_TOKENS = 256


@dataclass(frozen=True, slots=True)
class IonosRerankingBudget:
    """Conservatively estimate query/document pair token costs."""

    text_counter: TextTokenCounter
    image_metadata: ImageMetadataCache

    def count_pairs(
        self,
        query: MultimodalContent,
        documents: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
    ) -> list[int]:
        """Count each candidate with its repeated query and native images."""
        text_costs = self.text_counter.count_texts(
            [query.text or '', *(document.text or '' for document in documents)]
        )
        query_text_cost = text_costs[0]
        counts: list[int] = []
        for document, document_text_cost in zip(
            documents, text_costs[1:], strict=True
        ):
            cost = query_text_cost + document_text_cost + IONOS_FRAMING_TOKENS
            for position in (*query.image_positions, *document.image_positions):
                metadata = self.image_metadata.measure(content_images[position])
                cost += ((metadata.width + 31) // 32) * (
                    (metadata.height + 31) // 32
                )
            if cost > IONOS_MAX_PAIR_TOKENS:
                raise InputBudgetExceeded(
                    'IONOS reranking pair exceeds 32,768 estimated tokens'
                )
            counts.append(cost)
        return counts
