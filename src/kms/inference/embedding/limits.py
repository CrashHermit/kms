"""Published hard limits for Voyage multimodal embeddings."""

from dataclasses import dataclass, field

from kms.core.embedding import EmbeddingRequest
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.media.metadata import ImageMetadataCache

VOYAGE_MAX_INPUT_TOKENS = 32_000
VOYAGE_MAX_TOTAL_TOKENS = 320_000
VOYAGE_MAX_INPUTS = 1_000
VOYAGE_MAX_IMAGE_PIXELS = 16_000_000
VOYAGE_MAX_IMAGE_BYTES = 20_000_000
VOYAGE_EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True, slots=True)
class VoyageEmbeddingLimits:
    """Enforce Voyage's published hard item, request, and image constraints."""

    image_metadata: ImageMetadataCache = field(
        default_factory=ImageMetadataCache
    )

    def validate_items(
        self, request: EmbeddingRequest, costs: list[int]
    ) -> None:
        """Reject hard item/image violations before image loading or transport."""
        for material, cost in zip(request.materials, costs, strict=True):
            if cost > VOYAGE_MAX_INPUT_TOKENS:
                raise InputBudgetExceeded(
                    'Voyage embedding input exceeds 32,000 estimated tokens'
                )
            for image in material.images:
                metadata = self.image_metadata.measure(image)
                if metadata.width * metadata.height > VOYAGE_MAX_IMAGE_PIXELS:
                    raise InputBudgetExceeded(
                        'Voyage image exceeds 16,000,000 pixels'
                    )
                if metadata.byte_count > VOYAGE_MAX_IMAGE_BYTES:
                    raise InputBudgetExceeded(
                        'Voyage image exceeds 20,000,000 encoded bytes'
                    )

    @staticmethod
    def fits_batch(*, input_count: int, token_count: int) -> bool:
        """Check inclusive provider count and aggregate-token request limits."""
        return (
            input_count <= VOYAGE_MAX_INPUTS
            and token_count <= VOYAGE_MAX_TOTAL_TOKENS
        )
