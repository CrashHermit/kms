"""DSPy module for judging visual seams."""

import dspy

from kms2.core.model.source import SourceBlock


class ImageSeamSignature(dspy.Signature):
    """
    Decide whether two supplied images are one visual object split by a page
    seam. Return TRUE only when the bottom of the first image continues into
    the top of the second image. Return FALSE for two separate figures,
    unrelated images, or images where continuation is uncertain. Do not
    describe, transcribe, rewrite, stitch, or otherwise modify either image.
    """

    top_image: dspy.Image = dspy.InputField(
        description='The visual asset at the bottom of the top page.'
    )
    bottom_image: dspy.Image = dspy.InputField(
        description='The visual asset at the top of the bottom page.'
    )
    is_continuation: bool = dspy.OutputField(
        description='True only when both images are parts of one split visual object.'
    )


class ImageSeamJudgeModule(dspy.Module):
    """Make one boolean judgment about whether two images form one visual."""

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ImageSeamSignature)
        self.predictor.set_lm(language_model)

    @staticmethod
    def _inputs(
        *, top_block: SourceBlock, bottom_block: SourceBlock
    ) -> dict[str, dspy.Image]:
        return {
            'top_image': dspy.Image(url=top_block.assets[0].path),
            'bottom_image': dspy.Image(url=bottom_block.assets[0].path),
        }

    def forward(
        self, *, top_block: SourceBlock, bottom_block: SourceBlock
    ) -> bool:
        """Judge one adjacent image pair synchronously."""
        prediction = self.predictor(
            **self._inputs(top_block=top_block, bottom_block=bottom_block)
        )
        return prediction.is_continuation

    async def aforward(
        self, *, top_block: SourceBlock, bottom_block: SourceBlock
    ) -> bool:
        """Judge one adjacent image pair asynchronously."""
        prediction = await self.predictor.acall(
            **self._inputs(top_block=top_block, bottom_block=bottom_block)
        )
        return prediction.is_continuation
