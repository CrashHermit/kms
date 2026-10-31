"""DSPy module for source-faithful proposition extraction."""

import dspy

from kms2.core.model.source_extraction import (
    SourceExtractionInput,
)
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourcePropositionCandidate,
)


class SourcePropositionExtractionSignature(dspy.Signature):
    r"""Extract complete, independently meaningful claims from target blocks.

    Preserve conditions, exceptions, negation, quantifiers, formulas, domains,
    and defining objects. Resolve pronouns, demonstratives, source-local labels,
    and notation references only to explicitly supported names or definitions
    so each proposition stands alone. A symbol's name alone is not enough:
    include the supported object type, domain, and assumptions needed to
    interpret every occurrence of that symbol. Expand local setup assumptions
    into each dependent claim rather than emitting the setup separately.
    Never rely on another output proposition or surrounding book text to
    supply missing conditions.

    Target blocks carry zero-based local positions. Return exactly one
    source_position per proposition, selecting the target block that states the
    claim. Other target blocks and before/after context may resolve references
    and defining assumptions, but cannot contribute an independent claim to
    that owner. If the same claim is stated in different target blocks, return
    a separate occurrence for each source_position. Within one block, emit each
    distinct claim once.
    Do not invent an antecedent or mathematical assumption absent from the
    input. Omit claims that cannot satisfy this evidence contract. Preserve
    useful definitions, theorems, and properties without a derivation.
    Decide whether the source ASSERTS a claim, rather than asking the reader to
    construct, find, sketch, assume, or test something. Properties listed under
    such a task remain task requirements even when they occur in a later block
    or after a page break. Do not turn requested conditions into existence
    assertions, remove the task frame to present givens as general facts, or
    solve a task to manufacture a claim. Keep genuine source-asserted general
    facts, including explanatory hints embedded in an exercise, and genuine
    asserted existence results distinct from requested constructions.
    Classify clauses, not whole blocks: an exercise list can contain both
    unanswered questions and an explicitly asserted general rule in a hint.
    Extract that rule with its own conditions while omitting the surrounding
    questions and requested conclusions; do not discard the whole mixed block.
    A stated conditional algebraic identity in a hint is an asserted rule, not
    an instruction to derive it. Preserve both sign/case conditions. Its
    argument has the role given by that identity; a reused letter elsewhere in
    the exercise does not turn the rule into a numerical answer.
    Notation introductions and local setup declarations are not reusable
    propositions. Do not extract those declarations, exercise requests,
    intermediate arithmetic, or instance-specific numerical answers. Entity
    matches, proofs, and arguments are not admission requirements.
    Write all mathematics in `$...$` or `$$...$$`, preserving the supported
    notation and values. Do not use alternate math delimiters.
    """

    request: SourceExtractionInput = dspy.InputField()
    propositions: list[SourcePropositionCandidate] = dspy.OutputField(
        description='Self-contained claims, each with its one owning target source_position.'
    )


class SourcePropositionExtractorModule(dspy.Module):
    """Run proposition extraction for one UUID-free request."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, request: SourceExtractionInput
    ) -> list[SourcePropositionCandidate]:
        """Extract proposition candidates synchronously."""
        return self.predictor(request=request).propositions

    async def aforward(
        self, *, request: SourceExtractionInput
    ) -> list[SourcePropositionCandidate]:
        """Extract proposition candidates asynchronously."""
        return (await self.predictor.acall(request=request)).propositions
