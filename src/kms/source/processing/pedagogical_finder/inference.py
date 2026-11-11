"""DSPy modules for discovering pedagogical source components."""

import dspy

from kms.source.context import SourceBlockContext
from kms.source.processing.pedagogical_finder.models import (
    PedagogicalBoundaryDecision,
    PedagogicalClassificationInput,
    PedagogicalStartDecision,
)


class PedagogicalStartRouterSignature(dspy.Signature):
    r"""
    Use only images referenced by each item's target and context blocks; never
    borrow native visuals from another candidate.
    Classify each item's target_block independently as the possible start of a
    pedagogical component. The before and after lists are context only. Items
    are independent; never use one item as context for another.

    Return True for a labelled or numbered definition, theorem, proposition,
    lemma, corollary, axiom, law, model, rule, principle, worked example,
    exercise, problem, or prescribed procedure. A procedure lead-in and its
    numbered steps form one component.

    Return False for headings, ordinary narrative, shared exercise
    instructions, continuations, dangling OCR fragments, isolated answer
    choices, formatting artifacts, and material already owned by an earlier
    component. Do not merge distinct base numbers or attach unrelated nearby
    text.
    """

    items: list[PedagogicalClassificationInput] = dspy.InputField(
        description=(
            'Indexed targets with independent before/after context; items are '
            'not context for one another.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced only by each item’s target and context.'
    )
    decisions: list[PedagogicalStartDecision] = dspy.OutputField(
        description='One pedagogical-start decision for each input position.'
    )


class PedagogicalStartRouterModule(dspy.Module):
    """Route batches of source blocks that begin pedagogical components."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        items: list[PedagogicalClassificationInput],
    ) -> list[PedagogicalStartDecision]:
        """Classify independent pedagogical-start candidates synchronously."""
        return self.predictor(items=items).decisions

    async def aforward(
        self,
        *,
        items: list[PedagogicalClassificationInput],
    ) -> list[PedagogicalStartDecision]:
        """Classify independent pedagogical-start candidates asynchronously."""
        prediction = await self.predictor.acall(items=items)
        return prediction.decisions


class PedagogicalBoundaryRouterSignature(dspy.Signature):
    r"""
    Use only images referenced by this item's candidate and context plus the
    shared start block; never borrow another candidate's visual evidence.
    Classify each item's target_block relative to the pedagogical component
    that begins at start_block. Each item's before and after lists are context
    only; candidates are independent and never context for one another.

    Return False only when the candidate continues the anchored definition,
    theorem, proposition, lemma, corollary, axiom, law, model, rule, principle,
    worked example, exercise, problem, or prescribed procedure. Procedure steps
    and their subparts remain in that procedure. Return True when the candidate
    starts a distinct component or is unrelated material.

    Page numbers and repeated running titles are page furniture, not semantic
    boundaries. Return False across them so scanning continues through the
    contiguous span; layout blocks may remain members of that span without
    being semantic content. A genuine new section introducing different
    material, or the next distinct component, is a boundary.

    This is an exclusive boundary decision: a candidate at a boundary is not
    part of the anchored component. Do not consume it. Return one decision per
    indexed item.
    """

    start_block: SourceBlockContext = dspy.InputField(
        description='The first source block in one pedagogical component.'
    )
    items: list[PedagogicalClassificationInput] = dspy.InputField(
        description=(
            'Indexed boundary targets with independent before/after context; '
            'items are not context for one another.'
        )
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced only by this batch’s evidence.'
    )
    decisions: list[PedagogicalBoundaryDecision] = dspy.OutputField(
        description='One exclusive-boundary decision for each input position.'
    )


class PedagogicalBoundaryRouterModule(dspy.Module):
    """Route batches of exclusive pedagogical-boundary candidates."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[PedagogicalClassificationInput],
    ) -> list[PedagogicalBoundaryDecision]:
        """Classify exclusive component boundaries synchronously."""
        prediction = self.predictor(start_block=start_block, items=items)
        return prediction.decisions

    async def aforward(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[PedagogicalClassificationInput],
    ) -> list[PedagogicalBoundaryDecision]:
        """Classify exclusive component boundaries asynchronously."""
        prediction = await self.predictor.acall(
            start_block=start_block,
            items=items,
        )
        return prediction.decisions
