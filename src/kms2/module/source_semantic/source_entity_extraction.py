"""DSPy module for independent source entity extraction."""

import dspy

from kms2.core.model.source_extraction import SourceExtractionInput
from kms2.core.model.source_semantic.source_entity_extraction import (
    SourceEntityCandidate,
)


class SourceEntityExtractionSignature(dspy.Signature):
    r"""Extract explicitly discussed entities from target blocks.

    Include meaningful concepts, named results, structures, algorithms,
    objects, and contextually meaningful named people or organizations.
    Each target block has a zero-based local position. Return one source_position
    identifying the target block where each entity is mentioned or discussed.
    Resolve references using other supplied blocks, but never select reference
    context as ownership or emit unrelated context-only entities.
    Consolidate multiple mentions of the same entity within one target block,
    including supported aliases, into one occurrence. If the entity is mentioned
    in another target block, return a separate occurrence for that source_position.
    Do not emit every scalar or notation token as a global identity, or entities
    introduced only by outside knowledge. A meaningful concept heading or caption
    may yield an entity even when it yields no proposition. This is per-block
    occurrence extraction, not cross-block deduplication or description synthesis.
    """

    request: SourceExtractionInput = dspy.InputField()
    entities: list[SourceEntityCandidate] = dspy.OutputField(
        description='One occurrence per entity per target block, with its owning source_position.'
    )


class SourceEntityExtractorModule(dspy.Module):
    """Run entity extraction for one UUID-free request."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceExtractionInput
    ) -> list[SourceEntityCandidate]:
        """Extract entity candidates synchronously."""
        return self.predictor(request=request).entities

    async def aforward(
        self, *, request: SourceExtractionInput
    ) -> list[SourceEntityCandidate]:
        """Extract entity candidates asynchronously."""
        return (await self.predictor.acall(request=request)).entities
