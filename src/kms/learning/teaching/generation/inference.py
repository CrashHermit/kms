"""Generate one grounded explanation for a supplied learning target."""

import dspy

from kms.learning.teaching.generation.models import (
    TeachingDraft,
    TeachingGenerationRequest,
)


class TeachingGenerationSignature(dspy.Signature):
    """Teach one useful connection, example, or repair for the objective.

    Start from the learner's demonstrated understanding, preserve uncertainty,
    and do not claim understanding merely because an explanation is presented.
    Address cited evidence rather than inventing a misconception. For a
    solution target, explain that specific problem's valid solution. Use native
    images only by exact catalog index and ground all claims only in supplied
    knowledge or source material. Do not create an assessment task or search
    for sources. Grounding references must be drawn from supplied proposition,
    argument, and evidence-unit UUIDs.
    """

    request: TeachingGenerationRequest = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    teaching: TeachingDraft = dspy.OutputField()


class TeachingGenerationModule(dspy.Module):
    """Call an injected predictor for typed teaching drafts."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: TeachingGenerationRequest) -> TeachingDraft:
        return self.predictor(request=request).teaching

    async def aforward(
        self, *, request: TeachingGenerationRequest
    ) -> TeachingDraft:
        return (await self.predictor.acall(request=request)).teaching
