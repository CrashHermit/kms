"""DSPy modules for constructing statements and procedures."""

import dspy

from kms.core.model.source_pedagogical.pedagogical import (
    PedagogicalComponentInput,
    PedagogicalRoleDecision,
    ProcedurePartitionDecision,
    StatementPartitionDecision,
)


class PedagogicalRoleSignature(dspy.Signature):
    r"""
    Image positions are local to each member block. Use only images referenced
    by this component and never borrow images from another batch item.
    Judge two roles for each independent complete pedagogical component.
    Outer position addresses a component; members have their own zero-based
    positions local to that component. Other items are not evidence.

    has_statement — True when the component STATES something. It says that
    something is so, or asks for something to be done: a claim, a definition,
    or a problem posed to the reader. A clear unit label asserts that the text
    is a pedagogical unit; classify the full component based on what follows.

    has_procedure — True when the component WORKS something out: a proof, a
    solution, a derivation, or a worked calculation that resolves a statement.
    Signs of working include applying operations or results, computing, and
    drawing a conclusion.

    WORKING IS NOT ONLY ALGEBRA. Text that RESOLVES a statement is a procedure
    even when it manipulates no symbols at all: exhibiting an answer,
    analysing the posed case or figure, verifying, or justifying. Ask "does
    this text work out what came before it?" — not "does it contain
    equations?".

    A bare expression, equation, number, or formula in an exercise component
    states the task; notation alone is not visible working. Mark a procedure
    only when the source visibly works something out, gives an answer, or
    resolves an earlier statement.

    A standalone proof or solution may itself be a labeled source component.
    Judge what the component states and works out from its own content; do not
    infer a statement merely from a missing label.

    Judge the component in front of you on its own terms. A component that is
    neither a statement nor a procedure has both flags False and is skipped.
    A component that states something and then works it out has both flags True.

    Return one decision for every item, preserving its outer position.
    Return flags only; never rewrite, solve, or combine components.
    """

    items: list[PedagogicalComponentInput] = dspy.InputField(
        description='Independent complete components with component-local member indexes.'
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced by these components only.'
    )
    decisions: list[PedagogicalRoleDecision] = dspy.OutputField(
        description='One indexed statement/procedure role decision per component.'
    )


class PedagogicalRoleModule(dspy.Module):
    """Classify a complete pedagogical component's roles."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        items: list[PedagogicalComponentInput],
        content_images: list[dspy.Image],
    ) -> list[PedagogicalRoleDecision]:
        """Classify component roles synchronously."""
        return self.predictor(
            items=items, content_images=content_images
        ).decisions

    async def aforward(
        self,
        *,
        items: list[PedagogicalComponentInput],
        content_images: list[dspy.Image],
    ) -> list[PedagogicalRoleDecision]:
        """Classify component roles asynchronously."""
        prediction = await self.predictor.acall(
            items=items, content_images=content_images
        )
        return prediction.decisions


class StatementPartitionSignature(dspy.Signature):
    r"""
    Independently select the STATEMENT portion of every complete component in
    items: the claim, definition, or task posed. It says that something is so,
    or asks for something to be done, without doing it. Other components are
    not evidence. Use only native images referenced by this component's member
    blocks; never borrow image material from another batched component.

    In a component that only states something, EVERY member is in the
    statement portion. In a component that ALSO works something out (a
    derivation, proof, solution, or calculation), the statement portion is the
    stating part only: the working members are NOT part of it, even though
    they sit in the same component.

    Return one indexed decision per component, preserving outer position.
    statement_positions address only that component's member positions, not
    packet positions. Every member belongs to the statement, procedure, or
    both. Overlap is allowed when one member both poses and works.

    """

    items: list[PedagogicalComponentInput] = dspy.InputField(
        description='Independent complete components with component-local member positions.'
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced by these components only.'
    )
    decisions: list[StatementPartitionDecision] = dspy.OutputField(
        description='Outer component position and its selected inner statement positions.'
    )


class StatementPartitionModule(dspy.Module):
    """Partition a complete component into statement members."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        items: list[PedagogicalComponentInput],
        content_images: list[dspy.Image],
    ) -> list[StatementPartitionDecision]:
        """Select statement members synchronously."""
        return self.predictor(
            items=items, content_images=content_images
        ).decisions

    async def aforward(
        self,
        *,
        items: list[PedagogicalComponentInput],
        content_images: list[dspy.Image],
    ) -> list[StatementPartitionDecision]:
        """Select statement members asynchronously."""
        prediction = await self.predictor.acall(
            items=items, content_images=content_images
        )
        return prediction.decisions


class ProcedurePartitionSignature(dspy.Signature):
    r"""
    Independently select the PROCEDURE portion of every complete component in
    items: the text that WORKS SOMETHING OUT. A proof, solution, derivation,
    or worked calculation resolves a statement. Signs include substituting,
    integrating, factoring, splitting into cases, computing, and concluding.
    Other components are not evidence. Use only native images referenced by
    this component's member blocks; never borrow another component's images.

    In a component that only works something out, EVERY member is in the
    procedure portion. In a component that ALSO states something, the
    procedure portion is the working part only: the posing or asserting
    members are NOT part of it.

    Return one indexed decision per component, preserving outer position.
    procedure_positions address only that component's member positions, not
    packet positions. Every member belongs to the statement, procedure, or
    both. Overlap is allowed when one member both poses and works.
    """

    items: list[PedagogicalComponentInput] = dspy.InputField(
        description='Independent complete components with component-local member positions.'
    )
    content_images: list[dspy.Image] = dspy.InputField(
        description='Native images referenced by these components only.'
    )
    decisions: list[ProcedurePartitionDecision] = dspy.OutputField(
        description='Outer component position and its selected inner working positions.'
    )


class ProcedurePartitionModule(dspy.Module):
    """Partition a complete component into procedure members."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        *,
        items: list[PedagogicalComponentInput],
        content_images: list[dspy.Image],
    ) -> list[ProcedurePartitionDecision]:
        """Select procedure members synchronously."""
        return self.predictor(
            items=items, content_images=content_images
        ).decisions

    async def aforward(
        self,
        *,
        items: list[PedagogicalComponentInput],
        content_images: list[dspy.Image],
    ) -> list[ProcedurePartitionDecision]:
        """Select procedure members asynchronously."""
        prediction = await self.predictor.acall(
            items=items, content_images=content_images
        )
        return prediction.decisions
