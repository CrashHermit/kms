r"""
Member partitioners — LLM passes that decide WHICH nodes of a PCF block form
each portion.

The role typer has already decided whether a block contains a statement, a
procedure, or both, and the hubs start with the WHOLE span as their members.
For a block with only one role that membership is final. For a block with
BOTH, the line between the statement portion (what poses or asserts) and the
procedure portion (what works it out) has to be found: the statement
partitioner selects the statement nodes, the procedure partitioner the
procedure nodes. Each is an LLM pass over the block's member nodes, mirroring
the old statement/procedure extractors — but it selects NODES, it never
transcribes text.

The gate is set membership: a statement whose block also has a procedure is a
both-block and gets partitioned; a single-role block never pays for a call.
The two selections may overlap; they must never both be empty for a both-block
(``partition_statements`` / ``partition_procedures`` keep the full span when
the model returns nothing usable, so a bad answer never deletes a hub's
members).
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import llm, models, recorder, state

logger = logging.getLogger(__name__)


class WindowMember(BaseModel):
    """One member node of a PCF block as the partitioner sees it."""

    position: int
    type: str
    content: str | None = None


# ============================================================================
# 1. Statement partitioner
# ============================================================================


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

    current_nodes: list[WindowMember] = dspy.InputField(
        description="The block's member nodes, in order, each with a local position and its type."
    )
    statement_positions: list[int] = dspy.OutputField(
        description='Positions of the nodes that form the STATEMENT portion — the text that poses or asserts.'
    )


class StatementPartitioner(dspy.Module):
    """Selects the statement-portion nodes of a both-block.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.partitioner = dspy.ChainOfThought(StatementPartitionSignature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, current_nodes: list[WindowMember]) -> list[int]:
        """Select the statement-portion positions of one block.

        Args:
            current_nodes: The block's member nodes, each with a local position.

        Returns:
            The positions of the statement-portion nodes.
        """
        result = await self.partitioner.acall(current_nodes=current_nodes)
        recorder.record_example(
            'statement_partitioner', {'current_nodes': current_nodes}, result
        )
        positions = list(result.statement_positions or [])
        logger.debug(
            'statement partitioner: %d node(s) -> %d statement position(s)',
            len(current_nodes),
            len(positions),
        )
        return positions

    def forward(self, current_nodes: list[WindowMember]) -> list[int]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(current_nodes))


# ============================================================================
# 2. Procedure partitioner
# ============================================================================


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

    current_nodes: list[WindowMember] = dspy.InputField(
        description="The block's member nodes, in order, each with a local position and its type."
    )
    procedure_positions: list[int] = dspy.OutputField(
        description='Positions of the nodes that form the PROCEDURE portion — the text that works something out.'
    )


class ProcedurePartitioner(dspy.Module):
    """Selects the procedure-portion nodes of a both-block.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.partitioner = dspy.ChainOfThought(ProcedurePartitionSignature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, current_nodes: list[WindowMember]) -> list[int]:
        """Select the procedure-portion positions of one block.

        Args:
            current_nodes: The block's member nodes, each with a local position.

        Returns:
            The positions of the procedure-portion nodes.
        """
        result = await self.partitioner.acall(current_nodes=current_nodes)
        recorder.record_example(
            'procedure_partitioner', {'current_nodes': current_nodes}, result
        )
        positions = list(result.procedure_positions or [])
        logger.debug(
            'procedure partitioner: %d node(s) -> %d procedure position(s)',
            len(current_nodes),
            len(positions),
        )
        return positions

    def forward(self, current_nodes: list[WindowMember]) -> list[int]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(current_nodes))


# ============================================================================
# Helpers — window building and member selection
# ============================================================================


def _member_window(
    members: list[int], nodes_by_id: dict[int, models.ASTNode]
) -> list[WindowMember]:
    """The block's member nodes as window entries, in member order.

    Args:
        members: The hub's member node ids, in document order.
        nodes_by_id: The full node stream keyed by stable id.

    Returns:
        One ``WindowMember`` per resolvable member, position = index.
    """
    return [
        WindowMember(
            position=position,
            type=node.kind,
            content=node.content,
        )
        for position, node_id in enumerate(members)
        if (node := nodes_by_id.get(node_id)) is not None
    ]


def _selected_members(members: list[int], positions: list[int]) -> list[int]:
    """Map the LLM's window positions back to member node ids.

    Out-of-range positions are dropped; the result stays in member order.

    Args:
        members: The hub's member node ids.
        positions: The selected window positions.

    Returns:
        The selected member ids, in member order.
    """
    return [
        members[position]
        for position in positions
        if 0 <= position < len(members)
    ]


# ============================================================================
# Entry points — partition every both-block hub
# ============================================================================


async def partition_statements(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    nodes_by_id: dict[int, models.ASTNode],
    module: StatementPartitioner | None = None,
) -> None:
    """Narrow each both-block statement's members to its statement portion.

    A statement whose block also has a procedure is a both-block: the line is
    found by the LLM and ``members`` is replaced with the selected nodes.
    Single-role statements keep the whole block the role typer set. An empty
    selection is ignored — a bad answer must not delete a hub's members.

    Args:
        statements: The statement overlay, mutated in place.
        procedures: The procedure overlay, to identify both-blocks.
        nodes_by_id: The full node stream keyed by stable id.
        module: The statement partitioner. Created fresh if None.
    """
    module = module or StatementPartitioner()
    procedure_blocks = {tuple(procedure.block) for procedure in procedures}
    for statement in statements:
        if tuple(statement.block) not in procedure_blocks:
            continue
        window = _member_window(statement.members, nodes_by_id)
        positions = await module.aforward(window)
        selected = _selected_members(statement.members, positions)
        if selected:
            statement.members = selected

    logger.info(
        'statement partitioner: %d statement(s) scanned',
        len(statements),
    )


async def partition_procedures(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    nodes_by_id: dict[int, models.ASTNode],
    module: ProcedurePartitioner | None = None,
) -> None:
    """Narrow each both-block procedure's members to its procedure portion.

    A procedure whose block also has a statement is a both-block: the line is
    found by the LLM and ``members`` is replaced with the selected nodes.
    Single-role procedures keep the whole block the role typer set. An empty
    selection is ignored — a bad answer must not delete a hub's members.

    Args:
        statements: The statement overlay, to identify both-blocks.
        procedures: The procedure overlay, mutated in place.
        nodes_by_id: The full node stream keyed by stable id.
        module: The procedure partitioner. Created fresh if None.
    """
    module = module or ProcedurePartitioner()
    statement_blocks = {tuple(statement.block) for statement in statements}
    for procedure in procedures:
        if tuple(procedure.block) not in statement_blocks:
            continue
        window = _member_window(procedure.members, nodes_by_id)
        positions = await module.aforward(window)
        selected = _selected_members(procedure.members, positions)
        if selected:
            procedure.members = selected

    logger.info(
        'procedure partitioner: %d procedure(s) scanned',
        len(procedures),
    )


# ============================================================================
# LangGraph nodes
# ============================================================================


class StatementPartitionerNode:
    """Narrows each both-block statement's members to its statement portion.

    Args:
        module: The statement partitioner. Created fresh if None.
    """

    def __init__(self, module: StatementPartitioner | None = None) -> None:
        self.module = module or StatementPartitioner()

    async def run(self, state: state.State) -> dict:
        """Partition every both-block statement.

        Args:
            state: The pipeline state, holding the hub overlays and nodes.

        Returns:
            An empty update — the statements are mutated in place.
        """
        nodes = state.get('nodes', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        await partition_statements(
            state.get('statements', []),
            state.get('procedures', []),
            nodes_by_id,
            self.module,
        )
        return {}


class ProcedurePartitionerNode:
    """Narrows each both-block procedure's members to its procedure portion.

    Args:
        module: The procedure partitioner. Created fresh if None.
    """

    def __init__(self, module: ProcedurePartitioner | None = None) -> None:
        self.module = module or ProcedurePartitioner()

    async def run(self, state: state.State) -> dict:
        """Partition every both-block procedure.

        Args:
            state: The pipeline state, holding the hub overlays and nodes.

        Returns:
            An empty update — the procedures are mutated in place.
        """
        nodes = state.get('nodes', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        await partition_procedures(
            state.get('statements', []),
            state.get('procedures', []),
            nodes_by_id,
            self.module,
        )
        return {}
