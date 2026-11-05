"""DSPy module for source-grounded statement reconstruction."""

import dspy

from kms.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionItem,
    SourceStatementReconstructionOutput,
)


class SourceStatementReconstructionSignature(dspy.Signature):
    r"""Reconstruct every complete source-grounded statement independently.
    Use native images only when referenced by this item's target or instruction
    blocks; never borrow another item's visual evidence.

    Return readable Markdown with complete LaTeX. Preserve all requested
    subtasks, supplied data and hints, units, notation, definitions, negation,
    quantifiers, assumptions, domains, constraints, conditional alternatives,
    and required explanations or comparisons. Integrate every supplied
    governing instruction. Keep a theorem or claim as such, and a task as a
    task; do not solve an exercise or copy source solution material into its
    prompt, even when a source block belongs to both partitions. Resolve
    references only from this item's evidence. Use native figures referenced by
    its own blocks and include meaningful figure references without inventing
    a mathematical specification. Remove only headers, page numbers, answer
    blanks, navigational chatter, and conversational framing that
    carry no meaning. Never summarize, add facts, infer unsupported work, or
    borrow from another item. Preserve each packet-local position.
    """

    items: list[SourceStatementReconstructionItem] = dspy.InputField(
        description='Independent statement evidence packets with positions.'
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced only by each item’s evidence.'
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
        self,
        *,
        items: list[SourceStatementReconstructionItem],
        content_images: list[dspy.Image],
    ) -> list[SourceStatementReconstructionOutput]:
        """Reconstruct statement content synchronously."""
        return self.predictor(
            items=items, content_images=content_images
        ).reconstructions

    async def aforward(
        self,
        *,
        items: list[SourceStatementReconstructionItem],
        content_images: list[dspy.Image],
    ) -> list[SourceStatementReconstructionOutput]:
        """Reconstruct statement content asynchronously."""
        return (
            await self.predictor.acall(
                items=items, content_images=content_images
            )
        ).reconstructions
