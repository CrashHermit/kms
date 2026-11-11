"""Voyage multimodal embedding cost estimates."""

from dataclasses import dataclass

from kms.core.embedding import EmbeddingRequest
from kms.core.tokenization.text import TextTokenCounter
from kms.media.metadata import ImageMetadataCache

VOYAGE_FRAMING_TOKENS = 256

_QUERY_PREFIX = 'Represent the query for retrieving supporting documents: '
_DOCUMENT_PREFIX = 'Represent the document for retrieval: '


@dataclass(frozen=True, slots=True)
class VoyageEmbeddingEstimator:
    """Conservatively estimate Voyage token costs without enforcing capacity."""

    text_counter: TextTokenCounter
    image_metadata: ImageMetadataCache

    def count_inputs(self, request: EmbeddingRequest) -> list[int]:
        """Estimate one cost per material, including repeated image occurrences."""
        prefix = (
            _QUERY_PREFIX if request.input_type == 'query' else _DOCUMENT_PREFIX
        )
        text_costs = self.text_counter.count_texts(
            [prefix + (material.text or '') for material in request.materials]
        )
        costs: list[int] = []
        for material, text_cost in zip(
            request.materials, text_costs, strict=True
        ):
            image_cost = 0
            for image in material.images:
                metadata = self.image_metadata.measure(image)
                pixels = metadata.width * metadata.height
                image_cost += (max(50_000, pixels) + 559) // 560
            costs.append(text_cost + image_cost + VOYAGE_FRAMING_TOKENS)
        return costs
