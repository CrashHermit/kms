"""DSPy judgments for exact canonical proposition identity."""

import dspy

from kms2.core.model.knowledge.admission import (
    PropositionEquivalenceDecision,
    PropositionEquivalenceInput,
)


class PropositionEquivalenceSignature(dspy.Signature):
    r"""Judge whether candidates express the same proposition as the target.

    Return a decision for every candidate. Equivalence means the same complete
    proposition, not topical similarity, paraphrase with changed scope, or
    one-way implication. Compare assumptions, variable bindings, definitions,
    domains, quantifiers, modalities, and negation. Alpha-renaming of bound
    variables can preserve identity; changing an exercise-local function
    definition does not. A related or contradictory statement is not equivalent.
    """

    request: PropositionEquivalenceInput = dspy.InputField(
        description='Target claim and bounded claims with complete conditions.'
    )
    decisions: list[PropositionEquivalenceDecision] = dspy.OutputField(
        description='One explicit identity judgment for each candidate position.'
    )


class PropositionEquivalenceModule(dspy.Module):
    """Judge proposition identity through an injected predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: PropositionEquivalenceInput
    ) -> list[PropositionEquivalenceDecision]:
        """Judge candidates synchronously."""
        return self.predictor(request=request).decisions

    async def aforward(
        self, *, request: PropositionEquivalenceInput
    ) -> list[PropositionEquivalenceDecision]:
        """Judge candidates asynchronously."""
        prediction = await self.predictor.acall(request=request)
        return prediction.decisions
