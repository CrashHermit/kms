"""Compare proposition candidates for strict semantic equivalence."""

import dspy

from kms.knowledge.identity.models import IdentityDecision, IdentityRequest


class IdentitySignature(dspy.Signature):
    """Reuse only a supplied proposition with identical complete meaning.

    Compare the full claim, including its subject, predicate, scope, exclusions,
    conditions, modality, polarity, quantities, and qualifications. A related,
    broader, narrower, or merely compatible claim is not equivalent. Select a
    match only when it asserts precisely the same thing under the same scope.
    Select no match when none is identical. You may select only a UUID in the
    supplied matches; never invent or alter a UUID.
    """

    request: IdentityRequest = dspy.InputField()
    decision: IdentityDecision = dspy.OutputField()


class IdentityModule(dspy.Module):
    """Run typed identity decisions through an injected predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: IdentityRequest) -> IdentityDecision:
        return self.predictor(request=request).decision

    async def aforward(self, *, request: IdentityRequest) -> IdentityDecision:
        return (await self.predictor.acall(request=request)).decision
