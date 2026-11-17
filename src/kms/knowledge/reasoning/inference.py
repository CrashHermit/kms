"""DSPy module for reasoning from supplied propositions and arguments."""

import dspy

from kms.knowledge.reasoning.models import ReasoningCandidates, ReasoningRequest


class KnowledgeReasoningSignature(dspy.Signature):
    """Derive conclusions only when supplied premises explicitly support them.

    The understanding defines the question; it is not evidence. Use only the
    supplied proposition statements and complete arguments. For every proposed
    argument, include every proposition needed to justify the conclusion, and
    ensure those premises jointly entail it without relying on unstated facts,
    assumptions, or likely-but-unsupported rules. Do not add a conclusion whose
    proof would require a missing premise. Related claims are not a reason to
    infer a link. If no complete derivation is supported, return both candidate
    lists empty.

    Do not alter existing claims, restate them as new, or duplicate a supplied
    argument with identical premises and conclusion. New propositions need
    unique result-local ids. Argument references must point to a supplied
    proposition UUID or a new proposition id in this result.
    """

    request: ReasoningRequest = dspy.InputField()
    candidates: ReasoningCandidates = dspy.OutputField()


class KnowledgeReasoningModule(dspy.Module):
    """Run a typed reasoning prediction through an injected DSPy predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: ReasoningRequest) -> ReasoningCandidates:
        return self.predictor(request=request).candidates

    async def aforward(
        self, *, request: ReasoningRequest
    ) -> ReasoningCandidates:
        return (await self.predictor.acall(request=request)).candidates
