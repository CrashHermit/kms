"""DSPy module for source-local entity occurrence descriptions."""

import dspy

from kms2.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionInput,
)


class SourceEntityDescriptionSignature(dspy.Signature):
    r"""Describe one entity term in its supplied technical passage.

    Write a concise source-local gloss of what the noun phrase, object, or
    concept means in this passage. Preserve mathematical notation exactly, but
    distinguish a symbol or expression from the concept it denotes. When the
    passage explicitly identifies that meaning, use source-scoped wording such
    as "The symbol $v$ denotes velocity in this passage." If the meaning is not
    explicit, describe only the observed mathematical role and do not infer a
    referent. Use `$...$` for inline LaTeX and `$$...$$` for display math.
    Do not use alternate math delimiters. Do not create a general definition,
    merge synonyms, or invent facts. Describe only target_block; neighboring
    blocks are reference context.

    """

    request: SourceEntityDescriptionInput = dspy.InputField(
        description=(
            'Describe only target_block. context_before and context_after are '
            'reference context, not independent evidence.'
        )
    )
    description: str = dspy.OutputField(
        description='One concise, source-grounded local description.'
    )


class SourceEntityDescriptionModule(dspy.Module):
    """Generate one source-local description for an entity occurrence."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: SourceEntityDescriptionInput) -> str:
        """Describe one entity occurrence synchronously."""
        return self.predictor(request=request).description

    async def aforward(self, *, request: SourceEntityDescriptionInput) -> str:
        """Describe one entity occurrence asynchronously."""
        return (await self.predictor.acall(request=request)).description
