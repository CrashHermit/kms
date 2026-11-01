"""DSPy extraction of grounded, reusable propositions and arguments."""

import dspy

from kms2.core.model.knowledge.admission import (
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
    logistics, requests for answers, researcher briefs, hidden transcripts, or
    search-only results. Context blocks are candidates, not claims to exhaustively
    ingest; select only blocks that support a claim actually taught in response.
    Return empty lists when no eligible claims or arguments were taught.
    """

    request: KnowledgeExtractionInput = dspy.InputField(
        description='Completed visible response and the exact inspected evidence.'
    )
    evidence_images: list[dspy.Image] = dspy.InputField(
        description='Only inspected images mapped by the request block positions.'
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
        evidence_images: list[dspy.Image],
    ) -> KnowledgeDraft:
        """Extract a knowledge draft synchronously."""
        return self.predictor(
            request=request,
            evidence_images=evidence_images,
        ).result

    async def aforward(
        self,
        *,
        request: KnowledgeExtractionInput,
        evidence_images: list[dspy.Image],
    ) -> KnowledgeDraft:
        """Extract a knowledge draft asynchronously."""
        prediction = await self.predictor.acall(
            request=request,
            evidence_images=evidence_images,
        )
        return prediction.result
