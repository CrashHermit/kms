import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import content, llm, logs, models, recording, state

logger = logging.getLogger(__name__)


class WindowMember(BaseModel):
    position: int
    type: str
    content: str | None = None
    image_path: str | None = None


class Classify(dspy.Signature):
    r"""
    Judge two questions about a pedagogical block. Answer True or False
    for each. This is domain-neutral: the text may come from a math,
    physics, CS, or biology textbook.

    has_statement — True when the block STATES something. It says that
    something is so, or asks for something to be done: a claim, a
    definition, a problem posed to the reader. If the block opens with
    its own label naming it as a unit of the book — "Definition 2.5.1",
    "Theorem 3.4", "Example 6.7", "Exercise 12", or a bare leading
    number ("12.", "2.1.12") — it states something, whatever follows
    that label.

    has_procedure — True when the block WORKS something out: a proof, a
    solution, a derivation, a worked calculation that resolves what a
    block before it stated. Signs of working: substituting, integrating,
    factoring, splitting into cases, applying a named result, computing,
    concluding ("hence", "therefore", "so we get", "this completes the
    proof").

    WORKING IS NOT ONLY ALGEBRA. Text that RESOLVES a statement is a
    procedure even when it manipulates no symbols at all: exhibiting an
    answer ("Note that $y = 0$ is a solution. But another solution is
    the function ..."), analysing the posed case or figure, verifying
    or justifying ("$G_4$ is NOT a subgraph, because ..."). Ask
    "does this text work out what came before it?" — not "does it
    contain equations?".

    A COMPUTATION SESSION IS A PROCEDURE. Unlabelled transcript lines
    and their printed output — "sage: f = x^15 + 1",
    "sage: f.roots()", "[(12, 1), (10, 1), (4, 1)]", a shell or REPL
    session, a table of computed values — are the working of a block
    above them, so answer has_procedure = True.

    A derivation never carries a block label of its own — it either
    opens with a derivation marker ("Proof.", "Solution.") or is
    unlabelled text continuing from the block before it. For unlabelled
    text, never answer has_statement = True only because a marker word
    is missing.

    Judge the block in front of you on its own terms. Every block is
    at least one of these. A block that states something and then works
    it out has both flags True.
    """

    contents: content.ContentParts = dspy.InputField(
        description="The span's text and figures, in document order."
    )
    has_statement: bool = dspy.OutputField(
        description='True when the block states something (a claim, definition, theorem, example, exercise, or problem posed).'
    )
    has_procedure: bool = dspy.OutputField(
        description='True when the block works something out (a proof, solution, derivation, calculation, or computation session).'
    )


class RoleTyper(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.classify = dspy.Predict(Classify)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self, contents: content.ContentParts
    ) -> tuple[bool, bool]:
        result = await self.classify.acall(contents=contents)
        label = contents.content.render()
        if self._recorder:
            self._recorder.record('role_typer', {'contents': label}, result)
        has_statement = result.has_statement
        has_procedure = result.has_procedure
        if not has_statement and not has_procedure:
            raise ValueError(
                'Block classified as neither statement nor procedure. '
                f'Contents: {logs.elide(label)}'
            )
        logger.debug(
            'roles: has_statement=%s has_procedure=%s | from %r',
            has_statement,
            has_procedure,
            logs.elide(label),
        )
        return has_statement, has_procedure

    def forward(self, contents: content.ContentParts) -> tuple[bool, bool]:
        return asyncio.run(self.aforward(contents))


class StatementPartitionSignature(dspy.Signature):
    r"""
    You are given the member nodes of ONE pedagogical block from a textbook.
    Decide which of them form the STATEMENT portion — the part that STATES
    something: the claim, the definition, the task posed. It says that
    something is so, or asks for something to be done, without doing it.

    In a block that only states something, EVERY node is in the statement
    portion. In a block that ALSO works something out (a derivation — a proof,
    a solution, a calculation), the statement portion is the stating part
    only: the derivation nodes are NOT part of it, even though they sit in the
    same block.

    Return the positions of exactly the statement-portion nodes, over the
    given nodes ONLY. Every node is in the statement portion, the procedure
    portion, or both — never neither.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The block's member nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself.'
        )
    )
    statement_positions: list[int] = dspy.OutputField(
        description='Positions of the nodes that form the STATEMENT portion — the text that poses or asserts.'
    )


class StatementPartitioner(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.partitioner = dspy.Predict(StatementPartitionSignature)
        self.partitioner.demos = [
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowMember(
                            position=0,
                            type='paragraph',
                            content="**Exercise 1.2.1:** Sketch the slope field for $y' = e^{x-y}$.",
                        ),
                        WindowMember(
                            position=1,
                            type='paragraph',
                            content="**Exercise 1.2.2:** Sketch the slope field for $y' = x^2$.",
                        ),
                    ]
                ),
                statement_positions=[0, 1],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowMember(
                            position=0,
                            type='paragraph',
                            content="**Example 1.2.1:** Attempt to solve: $y' = \\frac{1}{x}, y(0) = 0$.",
                        ),
                        WindowMember(
                            position=1,
                            type='paragraph',
                            content='Integrate to find the general solution $y = \\ln |x| + C$.',
                        ),
                    ]
                ),
                statement_positions=[0],
            ).with_inputs('current_nodes'),
        ]
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, current_nodes: list[WindowMember]) -> list[int]:
        result = await self.partitioner.acall(
            current_nodes=content.labeled_content_parts(current_nodes)
        )
        if self._recorder:
            self._recorder.record(
                'statement_partitioner',
                {'current_nodes': current_nodes},
                result,
            )
        positions = list(result.statement_positions or [])
        logger.debug(
            'statement partitioner: %d node(s) -> %d statement position(s)',
            len(current_nodes),
            len(positions),
        )
        return positions

    def forward(self, current_nodes: list[WindowMember]) -> list[int]:
        return asyncio.run(self.aforward(current_nodes))


class ProcedurePartitionSignature(dspy.Signature):
    r"""
    You are given the member nodes of ONE pedagogical block from a textbook.
    Decide which of them form the PROCEDURE portion — the text that WORKS
    SOMETHING OUT: a proof, a solution, a derivation, a worked calculation. It
    resolves a block stated before it. Signs of working: substituting,
    integrating, factoring, splitting into cases, computing, concluding
    ("hence", "therefore", "so we get", "this completes the proof").

    In a block that only works something out, EVERY node is in the procedure
    portion. In a block that ALSO states something, the procedure portion is
    the working part only: the posing/asserting nodes are NOT part of it.

    Return the positions of exactly the procedure-portion nodes, over the
    given nodes ONLY. Every node is in the statement portion, the procedure
    portion, or both — never neither.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The block's member nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself.'
        )
    )
    procedure_positions: list[int] = dspy.OutputField(
        description='Positions of the nodes that form the PROCEDURE portion — the text that works something out.'
    )


class ProcedurePartitioner(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.partitioner = dspy.Predict(ProcedurePartitionSignature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, current_nodes: list[WindowMember]) -> list[int]:
        result = await self.partitioner.acall(
            current_nodes=content.labeled_content_parts(current_nodes)
        )
        if self._recorder:
            self._recorder.record(
                'procedure_partitioner',
                {'current_nodes': current_nodes},
                result,
            )
        positions = list(result.procedure_positions or [])
        logger.debug(
            'procedure partitioner: %d node(s) -> %d procedure position(s)',
            len(current_nodes),
            len(positions),
        )
        return positions

    def forward(self, current_nodes: list[WindowMember]) -> list[int]:
        return asyncio.run(self.aforward(current_nodes))


def _span_parts(
    span: list[int], nodes_by_id: dict[int, models.ASTNode]
) -> content.ContentParts:
    parts: list[content.TextPart | content.ImagePart] = []
    for node_id in span:
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        if node.type == 'image' and node.image_path:
            image = content.load_image(node.image_path)
            if image:
                parts.append(content.ImagePart(image=image))
        elif node.content and node.content.strip():
            parts.append(content.TextPart(text=node.content))
    return content.ContentParts(content=content.Content(parts=parts))


def _member_window(
    members: list[int], nodes_by_id: dict[int, models.ASTNode]
) -> list[WindowMember]:
    return [
        WindowMember(
            position=position,
            type=node.type,
            content=node.content,
            image_path=node.image_path,
        )
        for position, node_id in enumerate(members)
        if (node := nodes_by_id.get(node_id)) is not None
    ]


def _selected_members(members: list[int], positions: list[int]) -> list[int]:
    return [
        members[position]
        for position in positions
        if 0 <= position < len(members)
    ]


def _mark_statement(span: list[int]) -> models.Statement:
    return models.Statement(block=list(span), members=list(span))


def _mark_procedure(span: list[int]) -> models.Procedure:
    return models.Procedure(block=list(span), members=list(span))


async def _partition_both_block(
    statement: models.Statement,
    procedure: models.Procedure,
    nodes_by_id: dict[int, models.ASTNode],
    statement_partitioner: StatementPartitioner,
    procedure_partitioner: ProcedurePartitioner,
    gate: asyncio.Semaphore,
) -> None:
    window = _member_window(statement.members, nodes_by_id)
    async with gate:
        stmt_positions, proc_positions = await asyncio.gather(
            statement_partitioner.aforward(window),
            procedure_partitioner.aforward(window),
        )

    stmt_selected = _selected_members(statement.members, stmt_positions)
    proc_selected = _selected_members(procedure.members, proc_positions)
    if stmt_selected:
        statement.members = stmt_selected
    if proc_selected:
        procedure.members = proc_selected


async def build_hubs(
    spans: list[list[int]],
    nodes_by_id: dict[int, models.ASTNode],
    role_module: RoleTyper,
    statement_partitioner: StatementPartitioner | None = None,
    procedure_partitioner: ProcedurePartitioner | None = None,
    max_concurrency: int | None = None,
) -> tuple[list[models.Statement], list[models.Procedure]]:
    if not spans:
        logger.info('hub builder: no spans')
        return [], []

    gate = llm.gate(max_concurrency)

    async def _type_one(span: list[int]) -> tuple[bool, bool]:
        parts = _span_parts(span, nodes_by_id)
        if not parts.content.parts:
            return (False, False)
        async with gate:
            return await role_module.acall(parts)

    roles_by_span = await asyncio.gather(*(_type_one(span) for span in spans))

    statements: list[models.Statement] = []
    procedures: list[models.Procedure] = []
    both_blocks: list[tuple[models.Statement, models.Procedure]] = []

    for span, (has_statement, has_procedure) in zip(
        spans, roles_by_span, strict=True
    ):
        first_member_id = span[0]
        if first_member_id not in nodes_by_id:
            continue

        statement = _mark_statement(span) if has_statement else None
        procedure = _mark_procedure(span) if has_procedure else None

        if statement and procedure:
            both_blocks.append((statement, procedure))

        if statement:
            statements.append(statement)
        if procedure:
            procedures.append(procedure)
    if both_blocks:
        await asyncio.gather(
            *(
                _partition_both_block(
                    statement,
                    procedure,
                    nodes_by_id,
                    statement_partitioner,
                    procedure_partitioner,
                    gate,
                )
                for statement, procedure in both_blocks
            )
        )

    both_count = sum(
        1
        for s in statements
        if tuple(s.block) in {tuple(p.block) for p in procedures}
    )
    logger.info(
        'hub builder: %d span(s) -> %d statement(s), %d procedure(s), '
        '%d both-block(s)',
        len(spans),
        len(statements),
        len(procedures),
        both_count,
    )
    return statements, procedures


class HubBuilderNode:
    def __init__(
        self,
        role_module: RoleTyper,
        statement_partitioner: StatementPartitioner | None = None,
        procedure_partitioner: ProcedurePartitioner | None = None,
    ) -> None:
        self.role_module = role_module
        self.statement_partitioner = statement_partitioner
        self.procedure_partitioner = procedure_partitioner

    async def run(self, state: state.State) -> dict:
        nodes = state.get('nodes', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        statements, procedures = await build_hubs(
            state.get('spans', []),
            nodes_by_id,
            role_module=self.role_module,
            statement_partitioner=self.statement_partitioner,
            procedure_partitioner=self.procedure_partitioner,
        )
        return {'statements': statements, 'procedures': procedures}
