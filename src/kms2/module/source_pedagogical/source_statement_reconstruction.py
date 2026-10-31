"""DSPy module for source-grounded statement reconstruction."""

import dspy

from kms2.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionItem,
    SourceStatementReconstructionOutput,
)


class SourceStatementReconstructionSignature(dspy.Signature):
    r"""Reconstruct every complete source-grounded statement independently.

    Return readable Markdown with complete LaTeX. Preserve all requested
    subtasks, supplied data and hints, units, notation, definitions, negation,
    quantifiers, assumptions, domains, constraints, conditional alternatives,
    and required explanations or comparisons. Integrate every supplied
    governing instruction. Keep a theorem or claim as such, and a task as a
    task; do not solve an exercise or copy source solution material into its
    prompt, even when a source block belongs to both partitions. Resolve
    references only from this item's evidence. Preserve figure-relevant facts
    from image descriptions and include a meaningful figure reference without
    inventing a mathematical specification. Remove only headers, page numbers,
    empty answer blanks, navigational chatter, and conversational framing that
    carry no meaning. Never summarize, add facts, infer unsupported work, or
    borrow from another item. Preserve each packet-local position.
    """

    items: list[SourceStatementReconstructionItem] = dspy.InputField(
        description='Independent statement evidence packets with positions.'
    )
    reconstructions: list[SourceStatementReconstructionOutput] = (
        dspy.OutputField(
            description='One complete source-grounded reconstruction per item.'
        )
    )


class SourceStatementReconstructionModule(dspy.Module):
    """Reconstruct complete source-grounded statements in packets."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[SourceStatementReconstructionItem]
    ) -> list[SourceStatementReconstructionOutput]:
        """Reconstruct statement content synchronously."""
        return self.predictor(items=items).reconstructions

    async def aforward(
        self, *, items: list[SourceStatementReconstructionItem]
    ) -> list[SourceStatementReconstructionOutput]:
        """Reconstruct statement content asynchronously."""
        return (await self.predictor.acall(items=items)).reconstructions
