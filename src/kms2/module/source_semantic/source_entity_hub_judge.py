"""DSPy module for deciding source-local entity hub membership."""

import dspy

from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHubJudgeDecision,
    SourceEntityHubJudgeInput,
)


class SourceEntityHubJudgeSignature(dspy.Signature):
    r"""Judge identity equivalence between source-local entity occurrences.

    Return one boolean for each ordered pair. True only when evidence establishes
    that both occurrences denote the same entity. The owning block content and
    consumer-selected bounded source context are occurrence-local evidence; use them to
    distinguish repeated symbols such as f that have independently defined
    meanings in different exercises. Shared notation, topic, co-occurrence, or
    similar descriptions alone are not identity. False means the evidence does
    not establish sameness; never infer identity from a high similarity score.
    """

    requests: list[SourceEntityHubJudgeInput] = dspy.InputField(
        description='Ordered pairs with UUID-free owning-block content and local context.'
    )
    decisions: list[SourceEntityHubJudgeDecision] = dspy.OutputField(
        description='One indexed explicit identity-equivalence decision per pair.'
    )


class SourceEntityHubJudgeModule(dspy.Module):
    """Judge an ordered batch of entity occurrence pairs."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, requests: list[SourceEntityHubJudgeInput]
    ) -> list[SourceEntityHubJudgeDecision]:
        """Judge entity pairs synchronously."""
        prediction = self.predictor(requests=requests)
        return prediction.decisions

    async def aforward(
        self, *, requests: list[SourceEntityHubJudgeInput]
    ) -> list[SourceEntityHubJudgeDecision]:
        """Judge entity pairs asynchronously."""
        prediction = await self.predictor.acall(requests=requests)
        return prediction.decisions
