"""DSPy module for visual source-content correction."""

import dspy

from kms2.core.model.source_processing.content_correction import (
    ContentCorrectionInput,
    ContentCorrectionOutput,
)


class ContentCorrectorSignature(dspy.Signature):
    r"""Transcribe each indexed source crop faithfully, correcting OCR errors only.

    Each block is independent. Its zero-based position identifies exactly one
    crop at the same ordinal in block_crops. Use only that crop as evidence
    for its corrected_content; do not borrow text or answers from other blocks.
    Return one complete corrected transcription for every input position.

    Read the crop before consulting the OCR text. OCR is a potentially faulty
    aid, not a draft to copy: the crop is authoritative even when the OCR is
    mathematically plausible. Check every visible numeral, sign, superscript,
    and subscript against its own crop. Restore misread or missing visible
    characters, mathematical signs, labels, and text. Preserve wording, numbers,
    mathematical values, subparts, line order, and intentional answer spaces.
    Never fill in a blank, solve a problem, simplify an expression, infer
    missing answers, complete a cut-off sentence, or add explanatory text.
    Preserve a blank with its underline or placeholder, not a guessed value.

    Transcribe visible text even when the OCR content is absent. A crop that
    contains only a non-text figure has corrected_content null; do not describe
    or interpret the figure, and do not return OCR image-link placeholders as
    transcribed text. An empty crop likewise has no transcription. Mathematical
    notation may be represented in LaTeX without changing its content. This
    stage corrects transcription; representation formatting is a separate task.
    """

    blocks: list[ContentCorrectionInput] = dspy.InputField(
        description=(
            'Ordered source blocks with contiguous local positions, block types, '
            'and complete OCR transcriptions when available.'
        )
    )
    block_crops: list[dspy.Image] = dspy.InputField(
        description=(
            'Native source crop images, with crop ordinal matching each block '
            'position.'
        )
    )
    corrected_blocks: list[ContentCorrectionOutput] = dspy.OutputField(
        description=(
            'Complete corrected transcription for every input block, preserving '
            'its local position. Do not invent content.'
        )
    )


class ContentCorrectorModule(dspy.Module):
    """Run one non-recording full-block content correction prediction."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        blocks: list[ContentCorrectionInput],
        block_crops: list[dspy.Image],
    ) -> list[ContentCorrectionOutput]:
        """Correct a batch synchronously."""
        prediction = self.predictor(
            blocks=blocks,
            block_crops=block_crops,
        )
        return prediction.corrected_blocks

    async def aforward(
        self,
        *,
        blocks: list[ContentCorrectionInput],
        block_crops: list[dspy.Image],
    ) -> list[ContentCorrectionOutput]:
        """Correct a batch asynchronously."""
        prediction = await self.predictor.acall(
            blocks=blocks,
            block_crops=block_crops,
        )
        return prediction.corrected_blocks
