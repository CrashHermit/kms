"""Voyage multimodal embedding input limits."""

from dataclasses import dataclass
from typing import Literal

from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.core.tokenization.images import ImageMetadataCache
from kms.core.tokenization.text import TextTokenCounter

VOYAGE_MAX_INPUT_TOKENS = 32_000
VOYAGE_MAX_TOTAL_TOKENS = 320_000
VOYAGE_MAX_INPUTS = 1_000
VOYAGE_MAX_IMAGE_PIXELS = 16_000_000
VOYAGE_MAX_IMAGE_BYTES = 20_000_000
VOYAGE_FRAMING_TOKENS = 256

VOYAGE_EMBEDDING_DIMENSION = 1024

_QUERY_PREFIX = 'Represent the query for retrieving supporting documents: '
_DOCUMENT_PREFIX = 'Represent the document for retrieval: '


@dataclass(frozen=True, slots=True)
class VoyageEmbeddingBudget:
    """Conservatively estimate Voyage tokens and enforce published limits."""

    text_counter: TextTokenCounter
    image_metadata: ImageMetadataCache

    @property
    def text_token_limit(self) -> int:
        """Return text-only output capacity without loading tokenizers early."""
        document_prefix_tokens = self.text_counter.count_texts(
            [_DOCUMENT_PREFIX]
        )[0]
        return (
            VOYAGE_MAX_INPUT_TOKENS
            - VOYAGE_FRAMING_TOKENS
            - document_prefix_tokens
        )

    def count_inputs(
        self,
        materials: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        input_type: Literal['query', 'document'],
    ) -> list[int]:
        """Count conservative per-material costs, including repeated images."""
        prefix = _QUERY_PREFIX if input_type == 'query' else _DOCUMENT_PREFIX
        text_costs = self.text_counter.count_texts(
            [prefix + (material.text or '') for material in materials]
        )
        costs: list[int] = []
        for material, text_cost in zip(materials, text_costs, strict=True):
            image_cost = 0
            for position in material.image_positions:
                metadata = self.image_metadata.measure(content_images[position])
                pixels = metadata.width * metadata.height
                if pixels > VOYAGE_MAX_IMAGE_PIXELS:
                    raise InputBudgetExceeded(
                        'Voyage image exceeds 16,000,000 pixels'
                    )
                if metadata.byte_count > VOYAGE_MAX_IMAGE_BYTES:
                    raise InputBudgetExceeded(
                        'Voyage image exceeds 20,000,000 encoded bytes'
                    )
                image_cost += (max(50_000, pixels) + 559) // 560
            cost = text_cost + image_cost + VOYAGE_FRAMING_TOKENS
            if cost > VOYAGE_MAX_INPUT_TOKENS:
                raise InputBudgetExceeded(
                    'Voyage embedding input exceeds 32,000 estimated tokens'
                )
            costs.append(cost)
        return costs
