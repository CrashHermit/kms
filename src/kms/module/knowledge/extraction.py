"""DSPy extraction of grounded, reusable propositions and arguments."""

import dspy

from kms.core.model.knowledge.admission import (
    KnowledgeDraft,
    KnowledgeExtractionInput,
)


class KnowledgeExtractorSignature(dspy.Signature):
    r"""Extract only useful, grounded subject claims taught in the response.

    Write each proposition as a self-contained claim: preserve assumptions,
    notation and bindings, polarity, modalities, domains, and quantifiers. Resolve
    pronouns and local symbols using inspected context. Existing propositions may
    be premises only when explicitly read in this request. Every new proposition
    is eligible for support only from supplied blocks or grounded premises.

    Return one joint-premise reasoning record for each justified learner-facing
    inference, with exactly one conclusion. Do not infer derivation because claims
    are nearby. A conclusion is reusable as a later premise. Direct source support
    may be empty when grounded premise(s) support the conclusion.

    Do not extract learner mistakes/questions as facts, unsupported assertions,
    logistics, requests for answers, tool observations, hidden transcripts, or
    search-only results. Each response segment lists its own cited block,
    image, and proposition positions; do not support a segment from evidence
    assigned only to another segment. Context blocks are candidates, not claims
    to exhaustively ingest; select only blocks that support a claim actually
    taught in response. Citation association identifies the intended evidence,
    not proof that it entails a claim. Return empty lists when no eligible claims
    or arguments were taught.

    Each block's nested source_block.material.image_positions identifies only
    images inspected and assigned to that block. Use content_images only for the
    block that references each position; never borrow another block's images. A
    block with material.text=None has not had its text inspected.

    """

    request: KnowledgeExtractionInput = dspy.InputField(
        description=(
            'Ordered response segments and only the evidence explicitly '
            'associated with each segment.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description=(
            'Only inspected images referenced by nested '
            'source_block.material.image_positions.'
        )
    )
    result: KnowledgeDraft = dspy.OutputField(
        description='Grounded self-contained propositions and joint-premise steps.'
    )


class KnowledgeExtractorModule(dspy.Module):
    """Run a configured predictor for grounded knowledge extraction."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        request: KnowledgeExtractionInput,
        content_images: list[dspy.Image],
    ) -> KnowledgeDraft:
        """Extract a knowledge draft synchronously."""
        return self.predictor(
            request=request,
            content_images=content_images,
        ).result

    async def aforward(
        self,
        *,
        request: KnowledgeExtractionInput,
        content_images: list[dspy.Image],
    ) -> KnowledgeDraft:
        """Extract a knowledge draft asynchronously."""
        prediction = await self.predictor.acall(
            request=request,
            content_images=content_images,
        )
        return prediction.result
