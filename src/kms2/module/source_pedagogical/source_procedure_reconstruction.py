"""DSPy module for source-grounded procedure reconstruction."""

import dspy

from kms2.core.model.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionItem,
    SourceProcedureReconstructionOutput,
)


class SourceProcedureReconstructionSignature(dspy.Signature):
    r"""Reconstruct each complete source-grounded procedure independently.

    Return readable Markdown with complete LaTeX. Preserve the source's worked
    solution, derivation, or proof, including every supported step, ordering,
    intermediate equation, case distinction, condition, input, output,
    conclusion, and stopping criterion. Retain explanatory rationale needed
    to follow a proof or method. Use only this item's target blocks; never
    inherit statements or instructions, fill gaps, invent work, borrow from
    another item, or summarize away substantive material. Preserve figure
    facts supplied in descriptions without inventing unsupported details.
    Remove only headers, page numbers, empty answer blanks, navigational
    chatter, and conversational framing that carry no meaning. Preserve each
    packet-local position.
    """

    items: list[SourceProcedureReconstructionItem] = dspy.InputField(
        description='Independent procedure evidence packets with positions.'
    )
    reconstructions: list[SourceProcedureReconstructionOutput] = (
        dspy.OutputField(
            description='One complete source-grounded reconstruction per item.'
        )
    )


class SourceProcedureReconstructionModule(dspy.Module):
    """Reconstruct complete source-grounded procedures in packets."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, *, items: list[SourceProcedureReconstructionItem]
    ) -> list[SourceProcedureReconstructionOutput]:
        """Reconstruct procedure content synchronously."""
        return self.predictor(items=items).reconstructions

    async def aforward(
        self, *, items: list[SourceProcedureReconstructionItem]
    ) -> list[SourceProcedureReconstructionOutput]:
        """Reconstruct procedure content asynchronously."""
        return (await self.predictor.acall(items=items)).reconstructions
