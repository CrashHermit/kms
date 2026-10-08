"""DSPy module for source-faithful fact extraction."""

import dspy

from kms2.core.model.source_semantic.source_fact_extraction import (
    SourceAtomicFact,
    SourceFactExtractionInput,
)


class SourceFactExtractionSignature(dspy.Signature):
    r"""Extract explicit, durable facts asserted by target blocks.

    Extract one independently meaningful claim per fact. Keep its conditions,
    exceptions, negation, necessary participants, and other qualifiers
    attached. Prefer concise wording, but never remove information that changes
    the claim's meaning. Include reusable mathematical statements: definitions,
    identities, formulas, and explicitly stated conditions. An equation is not
    automatically an eligible fact. For mathematical relationships, identify
    what the equation describes and retain explicitly stated domains,
    conditions, and defining objects. Do not return a bare equation when its
    meaning depends on omitted text. Do not extract substitutions, intermediate
    arithmetic, or numerical answers computed for a particular exercise or
    worked example. For example, extract "Average velocity over [a,b], where
    a < b, is (s(b) - s(a)) / (b - a)." but return no facts for
    "(s(0.5) - s(0.49)) / (0.5 - 0.49) = -15.84." Keep useful setup facts
    even when they occur in an exercise or worked-example section.
    Target blocks are the only source evidence. Neighboring blocks may resolve
    references but are never independent evidence. Make each fact
    self-contained: resolve permitted references and avoid phrases such as
    "this equation", "the above", or "according to the source". Do not supply
    unstated assumptions or symbol definitions from outside knowledge.

    Do not require a fact to support a triplet; triplet decomposition is a
    later stage. Return no facts for headings, metadata, questions,
    instructions, exercise requests, calculation requests, substitutions,
    intermediate arithmetic, or instance-specific numerical answers. Return
    one concise fact per independent reusable claim.
    """

    request: SourceFactExtractionInput = dspy.InputField(
        description=(
            'target_blocks are the only source evidence. context_before and '
            'context_after may resolve references but cannot contribute facts. '
            'Preserve every qualifier and explicit mathematical relationship; '
            'triplet support is evaluated in a later stage.'
        )
    )
    facts: list[SourceAtomicFact] = dspy.OutputField(
        description=(
            'Source-faithful independent facts, including explicit mathematical '
            'claims, or an empty list.'
        )
    )


class SourceFactExtractorModule(dspy.Module):
    """Run the atomic source-fact extraction pass."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceFactExtractionInput
    ) -> list[SourceAtomicFact]:
        """Extract facts synchronously from one UUID-free local request."""
        return self.predictor(request=request).facts

    async def aforward(
        self, *, request: SourceFactExtractionInput
    ) -> list[SourceAtomicFact]:
        """Extract facts asynchronously from one UUID-free local request."""
        return (await self.predictor.acall(request=request)).facts
