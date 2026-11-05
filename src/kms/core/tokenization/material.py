"""Estimate LLM input material costs without prompt-template accounting."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from kms.core.content import ContentImage
from kms.core.tokenization.images import ImageMetadataCache
from kms.core.tokenization.text import TextTokenCounter


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate image tokens using 32-pixel patches and a 1.2 multiplier."""
    patches = ((width + 31) // 32) * ((height + 31) // 32)
    return (patches * 6 + 4) // 5


@dataclass(frozen=True, slots=True)
class MaterialTokenEstimator:
    """Estimate application payload tokens with cached native-image metadata."""

    text_counter: TextTokenCounter
    image_metadata: ImageMetadataCache = field(
        default_factory=ImageMetadataCache
    )

    def count_texts(self, texts: list[str]) -> list[int]:
        """Delegate text-only sizing to the consuming tokenizer."""
        return self.text_counter.count_texts(texts)

    def estimate(
        self,
        *,
        text: str | None = None,
        images: Sequence[ContentImage] = (),
    ) -> int:
        """Estimate payload tokens, excluding prompts and templates."""
        text_count = self.text_counter.count_texts([text])[0] if text else 0
        return text_count + self.estimate_images(images)

    def estimate_images(self, images: Sequence[ContentImage]) -> int:
        """Sum fixed LLM image estimates per supplied image occurrence."""
        return sum(
            estimate_image_tokens(metadata.width, metadata.height)
            for metadata in (
                self.image_metadata.measure(image) for image in images
            )
        )
