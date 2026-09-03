"""DSPy module for visual source-content correction."""

import dspy
from langgraph.types import Send

from kms2.core.model.source import OCRArtifact
from kms2.langgraph.source.state import SourceState


class ContentCorrectorSignature(dspy.Signature):
    """Restore OCR content so it matches the supplied source image."""

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped source image for this content block.'
    )
    block_type: str = dspy.InputField(
        description='The source block type, such as text or equation.'
    )
    content: str = dspy.InputField(
        description='The complete OCR transcription of the block.'
    )
    corrected_content: str = dspy.OutputField(
        description='The complete corrected block transcription.'
    )


class ContentCorrectorModule(dspy.Module):
    """Run one non-recording full-block content correction prediction."""

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ContentCorrectorSignature)
        self.predictor.set_lm(language_model)

    def forward(
        self,
        *,
        block_crop: dspy.Image,
        block_type: str,
        content: str,
    ) -> str:
        """Correct one OCR block synchronously."""
        prediction = self.predictor(
            block_crop=block_crop,
            block_type=block_type,
            content=content,
        )
        return prediction.corrected_content

    async def aforward(
        self,
        *,
        block_crop: dspy.Image,
        block_type: str,
        content: str,
    ) -> str:
        """Correct one OCR block asynchronously."""
        prediction = await self.predictor.acall(
            block_crop=block_crop,
            block_type=block_type,
            content=content,
        )
        return prediction.corrected_content


class ContentCorrectorNode:
    def dispatch(self, state: SourceState) -> dict:
        ocr_artifacts: list[OCRArtifact] = state['ocr_artifacts']
        sends: list[Send]
        for ocr_artifact in ocr_artifacts:
            send.append('content_corrector_worker', {'ocr_artifact': ocr_artifact})

        return sends

    async def worker(self, state: dict) -> dict:
        pass

    def collect(self, state: SourceState) -> dict:
        pass
