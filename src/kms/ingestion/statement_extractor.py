r"""
Statement extractor — fills ``content`` on each statement node from its
group text.

Concatenates the group's member node text into one string. An LLM pass extracts
the statement portion; for now the concatenation is deterministic.

The statements come in on their own ``statements`` channel and the members are
read out of ``nodes``; the two never mix. That separation is what keeps this
concatenation safe: the fused text lands on an entity beside the stream, so no
stage that walks ``nodes`` — the assembler above all — sees a group's text
twice.
"""

import logging

from kms.core import models, state

logger = logging.getLogger(__name__)


def group_text(members: list[models.ASTNode]) -> str:
    """The group's member node content as one string.

    Args:
        members: The group's member nodes, in document order.

    Returns:
        Their content joined blank-line separated.
    """
    return '\n\n'.join(
        member.content
        for member in members
        if member.content and member.content.strip()
    )


def extract_statement(
    statement_node: models.Statement,
    nodes_by_id: dict[int, models.ASTNode],
) -> None:
    """Fill ``content`` on a statement node from the group's member text.

    Concatenates member node text; the LLM pass will extract the statement
    portion from the full group.

    Args:
        statement_node: The statement node to fill, mutated in place.
        nodes_by_id: The full node stream keyed by stable id. It holds the
            group's members, never the statement itself.
    """
    member_ids = statement_node.statement_of or []
    members = [
        nodes_by_id[node_id] for node_id in member_ids if node_id in nodes_by_id
    ]
    statement_node.content = group_text(members) or None


# --- LangGraph node ---


class StatementExtractorNode:
    """Fills each statement node's ``content`` from its group text."""

    async def run(self, state: state.State) -> dict:
        """Fill ``content`` on every statement in the overlay.

        Args:
            state: The pipeline state, holding the node stream and the
                statement overlay.

        Returns:
            An empty update — the statement nodes are mutated in place.
        """
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        statements = state.get('statements', [])
        for statement in statements:
            extract_statement(statement, nodes_by_id)
        logger.info('statement extractor: %d statement(s)', len(statements))
        return {}
