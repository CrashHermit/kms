"""DSPy module for source-local entity occurrence descriptions."""

import dspy

from kms2.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionItem,
    SourceEntityDescriptionOutput,
)


class SourceEntityDescriptionSignature(dspy.Signature):
    r"""Independently describe each entity occurrence in its own technical passage.

    Write a concise source-local gloss of what the noun phrase, object, or
    concept means in this passage. Preserve mathematical notation exactly, but
    distinguish a symbol or expression from the concept it denotes. When the
    passage explicitly identifies that meaning, use source-scoped wording.
    Otherwise describe only the observed mathematical role without inferring
    a referent. Use `$...$` for inline LaTeX and `$$...$$` for display math.
    Do not use alternate math delimiters. Do not create a general definition,
    merge synonyms, or invent facts. Describe only target_blocks; neighboring
    blocks are reference context.

    Each item's request supplies its term, owning target_blocks and independent
    reference context. The same name may have different meanings in different
    items. Do not pool evidence, merge occurrences, or borrow another item's
    meaning. Return one description per item preserving its zero-based position.
    """

    items: list[SourceEntityDescriptionItem] = dspy.InputField(
        description='Independent entity requests with packet-local positions and own evidence.'
    )
    descriptions: list[SourceEntityDescriptionOutput] = dspy.OutputField(
        description='One indexed source-grounded local gloss per occurrence.'
    )


class SourceEntityDescriptionModule(dspy.Module):
    """Generate independent source-local glosses for entity occurrences."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[SourceEntityDescriptionItem]
    ) -> list[SourceEntityDescriptionOutput]:
        """Describe a packet of entity occurrences synchronously."""
        return self.predictor(items=items).descriptions

    async def aforward(
        self, *, items: list[SourceEntityDescriptionItem]
    ) -> list[SourceEntityDescriptionOutput]:
        """Describe a packet of entity occurrences asynchronously."""
        return (await self.predictor.acall(items=items)).descriptions
