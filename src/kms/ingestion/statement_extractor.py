r"""
Statement extractor — fills ``content`` on each statement node from its group text.

Concatenates the group's member node text into one string. An LLM pass extracts
the statement portion; for now the concatenation is deterministic.
"""

import logging

from kms.core import models, state

logger = logging.getLogger(__name__)


def group_text(members: list[models.ASTNode]) -> str:
    """The group's member node content as one string, blank-line separated."""
    return '\n\n'.join(
        member.content
        for member in members
        if member.content and member.content.strip()
    )


def extract_statement(
    statement_node: models.StatementNode,
    nodes_by_id: dict[int, models.ASTNode],
) -> None:
    """Fill ``content`` on a statement node from the group's member text.

    Concatenates member node text; the LLM pass will extract the statement
    portion from the full group.
    """
    member_ids = statement_node.statement_of or []
    members = [nodes_by_id[i] for i in member_ids if i in nodes_by_id]
    statement_node.content = group_text(members) or None


# --- LangGraph node ---


class StatementExtractorNode:
    """Fills each statement node's ``content`` from its group text."""

    async def run(self, state: state.State) -> dict:
        """Fill ``content`` on every statement node."""
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        for sid in state.get('statement_ids', []):
            if sid in nodes_by_id:
                extract_statement(nodes_by_id[sid], nodes_by_id)
        logger.info(
            'statement extractor: %d statement(s)',
            len(state.get('statement_ids', [])),
        )
        return {}
