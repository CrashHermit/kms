"""DSPy module for judging source-local proposition equivalence."""

import dspy

from kms2.core.model.source_semantic.source_proposition_hub import (
    SourcePropositionHubJudgeDecision,
    SourcePropositionHubJudgeInput,
)


class SourcePropositionHubJudgeSignature(dspy.Signature):
    r"""Judge whether two source propositions express the same claim.

    Allow paraphrase and consistent renaming of arbitrary variables with the
    same bindings and roles. Reject changes to domain, assumptions, quantifier
    scope, implication direction or strength, negation, constants, fixed
    functions, or symbol roles. Related, compatible, or shared-entity claims
    are not equivalent. Do not infer formal equivalence from wording alone.
    """

    requests: list[SourcePropositionHubJudgeInput] = dspy.InputField(
        description='Independent candidate proposition pairs to compare.'
    )
    decisions: list[SourcePropositionHubJudgeDecision] = dspy.OutputField(
        description='One indexed same-claim judgment for every pair.'
    )


class SourcePropositionHubJudgeModule(dspy.Module):
    """Judge an ordered batch of proposition pairs with one predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourcePropositionHubJudgeInput]
    ) -> list[SourcePropositionHubJudgeDecision]:
        """Judge proposition pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[SourcePropositionHubJudgeInput]
    ) -> list[SourcePropositionHubJudgeDecision]:
        """Judge proposition pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions
