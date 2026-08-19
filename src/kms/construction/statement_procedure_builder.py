"""Builds statement/procedure hub blocks from pedagogical spans."""

import asyncio
import logging

import dspy

from kms import config
from kms.core import (
    content,
    identity,
    llm,
    models,
    module,
    recording,
    state,
    walker,
)

logger = logging.getLogger(__name__)


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

    Judge the block in front of you on its own terms. A block that is
    neither a statement nor a procedure has both flags False and is
    skipped. A block that states something and then works it out has both
    flags True.
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


class RoleTyper(module.Module):
    """Decides whether a span states something and/or works it out."""

    signature = Classify
    record_name = 'role_typer'

    def encode(self, contents: content.ContentParts) -> dict:
        """Returns the role-typer signature kwargs."""
        return {'contents': contents}

    def decode(self, prediction, **inputs) -> tuple[bool, bool]:
        """Returns ``(has_statement, has_procedure)`` for one span."""
        return prediction.has_statement, prediction.has_procedure


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


class StatementPartitioner(module.Module):
    """Selects the nodes of a both-role block that form its statement."""

    signature = StatementPartitionSignature
    record_name = 'statement_partitioner'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content="**Exercise 1.2.1:** Sketch the slope field for $y' = e^{x-y}$.",
                        ),
                        walker.WindowNode(
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
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content="**Example 1.2.1:** Attempt to solve: $y' = \\frac{1}{x}, y(0) = 0$.",
                        ),
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            content='Integrate to find the general solution $y = \\ln |x| + C$.',
                        ),
                    ]
                ),
                statement_positions=[0],
            ).with_inputs('current_nodes'),
        ]

    def encode(self, current_nodes: list[walker.WindowNode]) -> dict:
        """Builds the partitioner-signature kwargs for one block."""
        return {'current_nodes': content.labeled_content_parts(current_nodes)}

    def decode(self, prediction, **inputs) -> list[int]:
        """Returns the statement-portion positions from the prediction."""
        return module.as_list(prediction.statement_positions)


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


class ProcedurePartitioner(module.Module):
    """Selects the nodes of a both-role block that form its procedure."""

    signature = ProcedurePartitionSignature
    record_name = 'procedure_partitioner'

    def encode(self, current_nodes: list[walker.WindowNode]) -> dict:
        """Builds the partitioner-signature kwargs for one block."""
        return {'current_nodes': content.labeled_content_parts(current_nodes)}

    def decode(self, prediction, **inputs) -> list[int]:
        """Returns the procedure-portion positions from the prediction."""
        return module.as_list(prediction.procedure_positions)


def _span_parts(
    span: list[int], nodes_by_id: dict[int, models.Node]
) -> content.ContentParts:
    """Builds the multimodal content of one span for the role typer."""
    parts: list[content.TextPart | content.ImagePart] = []
    for node_id in span:
        node = nodes_by_id.get(node_id)
        if node is None:
            raise ValueError(f'role span references unknown node id {node_id}')
        if node.type == 'image' and node.image_path:
            image = content.load_image(
                node.image_path,
                max_dim=config.get_settings().image.max_dim,
            )
            if image:
                parts.append(content.ImagePart(image=image))
        elif node.content and node.content.strip():
            parts.append(content.TextPart(text=node.content))
    return content.ContentParts(content=content.Content(parts=parts))


def _member_window(
    members: list[int], nodes_by_id: dict[int, models.Node]
) -> list[walker.WindowNode]:
    """Builds the ordered walker.WindowNode view of a block's members."""
    window: list[walker.WindowNode] = []
    for position, node_id in enumerate(members):
        node = nodes_by_id.get(node_id)
        if node is None:
            raise ValueError(
                f'partition span references unknown node id {node_id}'
            )
        window.append(
            walker.WindowNode(
                position=position,
                id=node.id,
                type=node.type,
                content=node.content,
                image_path=node.image_path,
            )
        )
    return window


def _selected_members(members: list[int], positions: list[int]) -> list[int]:
    """Picks member ids from a valid partitioner response."""
    if len(set(positions)) != len(positions):
        raise ValueError(
            f'partitioner returned duplicate positions: {positions}'
        )
    for position in positions:
        if not 0 <= position < len(members):
            raise ValueError(
                f'partitioner position {position} is invalid for '
                f'{len(members)} member(s)'
            )
    return [members[position] for position in positions]


def _mark_statement(span: list[int]) -> models.Statement:
    """Marks a span as a Statement with all members initially included."""
    return models.Statement(block=list(span), members=list(span))


def _mark_procedure(span: list[int]) -> models.Procedure:
    """Marks a span as a Procedure with all members initially included."""
    return models.Procedure(block=list(span), members=list(span))


async def _partition_both_block(
    statement: models.Statement,
    procedure: models.Procedure,
    nodes_by_id: dict[int, models.Node],
    statement_partitioner: StatementPartitioner,
    procedure_partitioner: ProcedurePartitioner,
    gate: asyncio.Semaphore,
) -> None:
    """Splits a both-role block's members between statement and procedure.

    Runs both partitioners on the same member window and narrows each
    model's members to the positions it selected.
    """
    window = _member_window(statement.members, nodes_by_id)
    async with gate:
        stmt_positions, proc_positions = await asyncio.gather(
            statement_partitioner.aforward(current_nodes=window),
            procedure_partitioner.aforward(current_nodes=window),
        )

    stmt_selected = _selected_members(statement.members, stmt_positions)
    proc_selected = _selected_members(procedure.members, proc_positions)
    statement.members = stmt_selected
    procedure.members = proc_selected


async def build_statement_procedure_hubs(
    spans: list[list[int]],
    nodes_by_id: dict[int, models.Node],
    role_module: RoleTyper,
    statement_partitioner: StatementPartitioner | None = None,
    procedure_partitioner: ProcedurePartitioner | None = None,
    max_concurrency: int | None = None,
    source: str | None = None,
) -> tuple[list[models.Statement], list[models.Procedure]]:
    """Builds Statement and Procedure models from pedagogical spans.

    Each span is typed by role; spans with both roles have their
    members partitioned between the statement and procedure portions.

    Args:
        spans: Member-id lists, one per pedagogical unit.
        nodes_by_id: The node stream keyed by stable id.
        role_module: The role-typing module.
        statement_partitioner: Optional partitioner for both-role spans.
        procedure_partitioner: Optional partitioner for both-role spans.
        max_concurrency: Cap on concurrent LLM calls.

    Returns:
        ``(statements, procedures)`` model lists.
    """
    if not spans:
        logger.info('hub builder: no spans')
        return [], []

    gate = llm.gate(max_concurrency)

    async def _type_one(span: list[int]) -> tuple[bool, bool]:
        """Types one span, skipping spans with no usable content."""
        parts = _span_parts(span, nodes_by_id)
        if not parts.content.parts:
            return (False, False)
        async with gate:
            return await role_module.acall(contents=parts)

    roles_by_span = await asyncio.gather(*(_type_one(span) for span in spans))

    statements: list[models.Statement] = []
    procedures: list[models.Procedure] = []
    both_blocks: list[tuple[models.Statement, models.Procedure]] = []

    for span, (has_statement, has_procedure) in zip(
        spans, roles_by_span, strict=True
    ):
        if not span:
            raise ValueError('role typer returned an empty pedagogical span')
        for member_id in span:
            if member_id not in nodes_by_id:
                raise ValueError(
                    f'role span references unknown node id {member_id}'
                )

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
        for statement in statements
        if tuple(statement.block)
        in {tuple(procedure.block) for procedure in procedures}
    )
    if source is not None:
        identity.assign_statement_procedure_ids(statements, procedures, source)

    logger.info(
        'hub builder: %d span(s) -> %d statement(s), %d procedure(s), '
        '%d both-block(s)',
        len(spans),
        len(statements),
        len(procedures),
        both_count,
    )
    return statements, procedures


class StatementProcedureBuilderNode:
    """Graph node that turns spans into statements and procedures."""

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
        """Builds statements and procedures from the state's spans.

        Args:
            state: Pipeline state with nodes and spans.

        Returns:
            ``statements`` and ``procedures`` updates for the state.
        """
        nodes = state.get('nodes', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        if not state.get('spans', []):
            return {'statements': [], 'procedures': []}
        source = state.get('source_key', '').strip()
        if not source:
            source_object = state.get('source')
            source = models.source_key(source_object) or ''
        if not source:
            raise ValueError('statement/procedure construction requires a source')
        statements, procedures = await build_statement_procedure_hubs(
            state.get('spans', []),
            nodes_by_id,
            role_module=self.role_module,
            statement_partitioner=self.statement_partitioner,
            procedure_partitioner=self.procedure_partitioner,
            source=source,
        )
        return {'statements': statements, 'procedures': procedures}
