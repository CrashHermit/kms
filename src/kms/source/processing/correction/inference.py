"""DSPy module for visual source-content correction."""

import dspy

from kms.source.processing.correction.models import (
    ContentCorrectionInput,
    ContentCorrectionOutput,
)


class ContentCorrectorSignature(dspy.Signature):
    r"""Correct each non-image block from its own crop, OCR, and positional context.

    Each block is independent. Its material carries OCR text and authoritative
    target crop. The context image preserves page layout; target_bounds
    identifies the target block within that image.
    Read the positional sequence as previous, target, next source blocks, using
    adjacent blocks only to complete text visibly cut off at a page edge when
    it is clearly the same target block. Never copy an independent neighboring
    block, transfer its content to this target, or change target block
    ownership. Do not use another target's crop or material. Return one result
    for every original zero-based position, including rejected blocks.

    Read the crop before consulting OCR. Check every visible numeral, sign,
    superscript, and subscript. Restore misread or missing visible characters,
    signs, labels, and text. Preserve wording, values, subparts, line order,
    and intentional answer spaces. Never solve, simplify, infer answers, or
    add explanation.
    Reject only clearly non-source interface controls, advertisements, or
    extraction artifacts. Keep uncertain material, titles, captions, citations,
    intentional blanks, exercises, and instructional discussion of ads, URLs,
    or interfaces. For a wholly irrelevant block, return keep=false,
    corrected_content=null, cleanup_instructions=[]. For a retained block,
    return a faithful corrected transcription including any mixed artifact.
    Put exact, narrow removal directions for those artifacts in
    cleanup_instructions; use [] for clean content. Null or blank text never
    signals deletion.

    Treat only visibly corrupted non-source URL/OCR fragments as artifacts.
    Never test network availability, guess a replacement destination, or remove
    intact source links or code examples. Preserve answer blanks. Visible text
    in a non-image crop must be transcribed; do not describe figures or
    transcribe image-link placeholders. Mathematical notation may use LaTeX
    without changing content.
    """

    blocks: list[ContentCorrectionInput] = dspy.InputField(
        description=(
            'Ordered source blocks with contiguous local positions, block types, '
            'target crop and OCR text, plus native context image and target '
            'bounds.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native target crops and context images.'
    )

    corrected_blocks: list[ContentCorrectionOutput] = dspy.OutputField(
        description=(
            'One keep decision and faithful corrected transcription for every '
            'input position, including rejected blocks; include exact narrow '
            'cleanup directions for retained mixed content.'
        )
    )


class ContentCorrectorModule(dspy.Module):
    """Run one non-recording full-block content correction prediction."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, blocks: list[ContentCorrectionInput]
    ) -> list[ContentCorrectionOutput]:
        """Correct a batch synchronously."""
        prediction = self.predictor(blocks=blocks)
        return prediction.corrected_blocks

    async def aforward(
        self, *, blocks: list[ContentCorrectionInput]
    ) -> list[ContentCorrectionOutput]:
        """Correct a batch asynchronously."""
        prediction = await self.predictor.acall(blocks=blocks)
        return prediction.corrected_blocks
