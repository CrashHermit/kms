"""DSPy module for source-image description enrichment."""

import dspy

from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source import SourceBlock


class ImageEnrichmentSignature(dspy.Signature):
    """Describe one logical image using its ordered assets and source context."""

    target_images: list[dspy.Image] = dspy.InputField(
        description='Ordered visual assets belonging to one logical image block.'
    )
    context_before: list[SourceBlockContext] = dspy.InputField(
        description='Source blocks immediately before the target image.'
    )
    context_after: list[SourceBlockContext] = dspy.InputField(
        description='Source blocks immediately after the target image.'
    )
    description: str = dspy.OutputField(
        description='A concise, source-grounded description of the logical image.'
    )


class ImageEnrichmentModule(dspy.Module):
    """Describe one source image from its visual assets and nearby context."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    @staticmethod
    def _inputs(
        *,
        source_block: SourceBlock,
        context_before: list[SourceBlockContext],
        context_after: list[SourceBlockContext],
    ) -> dict[str, object]:
        return {
            'target_images': [
                dspy.Image(url=asset.path) for asset in source_block.assets
            ],
            'context_before': context_before,
            'context_after': context_after,
        }

    def forward(
        self,
        *,
        source_block: SourceBlock,
        context_before: list[SourceBlockContext],
        context_after: list[SourceBlockContext],
    ) -> str:
        """Describe one source image synchronously."""
        prediction = self.predictor(
            **self._inputs(
                source_block=source_block,
                context_before=context_before,
                context_after=context_after,
            )
        )
        return prediction.description

    async def aforward(
        self,
        *,
        source_block: SourceBlock,
        context_before: list[SourceBlockContext],
        context_after: list[SourceBlockContext],
    ) -> str:
        """Describe one source image asynchronously."""
        prediction = await self.predictor.acall(
            **self._inputs(
                source_block=source_block,
                context_before=context_before,
                context_after=context_after,
            )
        )
        return prediction.description
