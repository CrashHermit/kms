"""DSPy modules for constructing statements and procedures."""

import dspy

from kms2.core.model import BlockType, PedagogicalMember, SourceBlockContext


def _member(position: int, content: str) -> PedagogicalMember:
    """Build one text-only demonstration member."""
    return PedagogicalMember(
        position=position,
        source_block=SourceBlockContext(
            block_type=BlockType.PARAGRAPH,
            content=content,
        ),
    )


def _set_demos(predictor: dspy.Module, demos: list[dspy.Example]) -> None:
    """Attach demos to the underlying DSPy predictor when wrapped."""
    getattr(predictor, 'predictor', predictor).demos = demos


class PedagogicalRoleSignature(dspy.Signature):
    r"""
    Judge two questions about one pedagogical component. Answer True or False
    for each. This is domain-neutral: the text may come from a math, physics,
    CS, or biology textbook.

    has_statement — True when the component STATES something. It says that
    something is so, or asks for something to be done: a claim, a definition,
    or a problem posed to the reader. If the component opens with its own
    label naming it as a unit of the book — "Definition 2.5.1", "Theorem 3.4",
    "Example 6.7", "Exercise 12", or a bare leading number ("12.", "2.1.12")
    — it states something, whatever follows that label.

    has_procedure — True when the component WORKS something out: a proof, a
    solution, a derivation, a worked calculation that resolves what a block
    before it stated. Signs of working: substituting, integrating, factoring,
    splitting into cases, applying a named result, computing, concluding
    ("hence", "therefore", "so we get", "this completes the proof").

    WORKING IS NOT ONLY ALGEBRA. Text that RESOLVES a statement is a procedure
    even when it manipulates no symbols at all: exhibiting an answer,
    analysing the posed case or figure, verifying, or justifying. Ask "does
    this text work out what came before it?" — not "does it contain
    equations?".

    A BARE EXPRESSION, EQUATION, NUMBER, OR FORMULA IS NOT BY ITSELF WORKING.
    In an exercise component, inputs such as "25 - 7", "5 · 6", "$x^2 + 1$",
    or "$f(x)=x^2$" are the content of a problem posed to the reader: set
    has_statement = True and has_procedure = False unless the component also
    contains visible working or an answer. Do not infer a result from the
    expression and do not treat mathematical notation alone as computation.

    A COMPUTATION SESSION IS A PROCEDURE only when the source shows the
    computation being carried out or its output. Unlabelled transcript lines
    and printed output, a shell or REPL session, or a table of computed values
    are the working of a component above them. A lone input expression is not.

    A derivation never carries a block label of its own — it either opens with
    a derivation marker ("Proof.", "Solution.") or is unlabelled text
    continuing from the component before it. For unlabelled text, never answer
    has_statement = True only because a marker word is missing.

    Judge the component in front of you on its own terms. A component that is
    neither a statement nor a procedure has both flags False and is skipped.
    A component that states something and then works it out has both flags True.
    """

    members: list[PedagogicalMember] = dspy.InputField(
        description='All members of one pedagogical component in source order.'
    )
    has_statement: bool = dspy.OutputField(
        description='True when the component states something: a claim, definition, theorem, example, exercise, or problem posed.'
    )
    has_procedure: bool = dspy.OutputField(
        description='True when the component works something out: a proof, solution, derivation, calculation, or computation session.'
    )


class PedagogicalRoleModule(dspy.Module):
    """Classify a complete pedagogical component's roles."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, members: list[PedagogicalMember]) -> tuple[bool, bool]:
        """Classify component roles synchronously."""
        prediction = self.predictor(members=members)
        return prediction.has_statement, prediction.has_procedure

    async def aforward(
        self, *, members: list[PedagogicalMember]
    ) -> tuple[bool, bool]:
        """Classify component roles asynchronously."""
        prediction = await self.predictor.acall(members=members)
        return prediction.has_statement, prediction.has_procedure


class StatementPartitionSignature(dspy.Signature):
    r"""
    You are given the member blocks of ONE pedagogical component from a
    textbook. Decide which of them form the STATEMENT portion — the part that
    STATES something: the claim, the definition, or the task posed. It says
    that something is so, or asks for something to be done, without doing it.

    In a component that only states something, EVERY member is in the
    statement portion. In a component that ALSO works something out (a
    derivation, proof, solution, or calculation), the statement portion is the
    stating part only: the working members are NOT part of it, even though
    they sit in the same component.

    Return the positions of exactly the statement-portion members, over the
    given members ONLY. Every member is in the statement portion, the
    procedure portion, or both — never neither.

    """

    members: list[PedagogicalMember] = dspy.InputField(
        description='The component members in source order. Positions are zero-based and local.'
    )
    statement_positions: list[int] = dspy.OutputField(
        description='Zero-based local positions of members that form the statement portion: text that poses or asserts.'
    )


class StatementPartitionModule(dspy.Module):
    """Partition a complete component into statement members."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

        _set_demos(
            predictor,
            [
                dspy.Example(
                    members=[
                        _member(
                            0,
                            "Exercise 1.2.1: Sketch the slope field for y' = e^(x-y).",
                        ),
                        _member(
                            1,
                            "Exercise 1.2.2: Sketch the slope field for y' = x^2.",
                        ),
                    ],
                    statement_positions=[0, 1],
                ).with_inputs('members'),
                dspy.Example(
                    members=[
                        _member(
                            0,
                            "Example 1.2.1: Attempt to solve y' = 1/x, y(0) = 0.",
                        ),
                        _member(
                            1,
                            'Integrate to find the general solution y = ln |x| + C.',
                        ),
                    ],
                    statement_positions=[0],
                ).with_inputs('members'),
            ],
        )

    def forward(self, *, members: list[PedagogicalMember]) -> list[int]:
        """Select statement members synchronously."""
        prediction = self.predictor(members=members)
        return prediction.statement_positions

    async def aforward(self, *, members: list[PedagogicalMember]) -> list[int]:
        """Select statement members asynchronously."""
        prediction = await self.predictor.acall(members=members)
        return prediction.statement_positions


class ProcedurePartitionSignature(dspy.Signature):
    r"""
    You are given the member blocks of ONE pedagogical component from a
    textbook. Decide which of them form the PROCEDURE portion — the text that
    WORKS SOMETHING OUT: a proof, a solution, a derivation, or a worked
    calculation. It resolves a component stated before it. Signs of working:
    substituting, integrating, factoring, splitting into cases, computing,
    concluding ("hence", "therefore", "so we get", "this completes the proof").

    In a component that only works something out, EVERY member is in the
    procedure portion. In a component that ALSO states something, the
    procedure portion is the working part only: the posing or asserting
    members are NOT part of it.

    Return the positions of exactly the procedure-portion members, over the
    given members ONLY. Every member is in the statement portion, the
    procedure portion, or both — never neither.
    """

    members: list[PedagogicalMember] = dspy.InputField(
        description='The component members in source order. Positions are zero-based and local.'
    )
    procedure_positions: list[int] = dspy.OutputField(
        description='Zero-based local positions of members that form the procedure portion: text that works something out.'
    )


class ProcedurePartitionModule(dspy.Module):
    """Partition a complete component into procedure members."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, members: list[PedagogicalMember]) -> list[int]:
        """Select procedure members synchronously."""
        prediction = self.predictor(members=members)
        return prediction.procedure_positions

    async def aforward(self, *, members: list[PedagogicalMember]) -> list[int]:
        """Select procedure members asynchronously."""
        prediction = await self.predictor.acall(members=members)
        return prediction.procedure_positions
