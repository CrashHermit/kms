r"""
Procedure extractor — fills ``content`` on each Procedure attached to a
statement.

Procedures are children of StatementNode.procedures (zero or one per
statement). The extractor reads the owning statement's group text and extracts
the procedure portion. For now the concatenation is deterministic; an LLM pass
will extract the procedure portion from the full group.
"""

import logging

from kms.core import models, state
from kms.ingestion import statement_extractor

logger = logging.getLogger(__name__)


def extract_procedures(
    statements: list[models.StatementNode],
    nodes_by_id: dict[int, models.ASTNode],
) -> None:
    """Fill ``content`` on every Procedure attached to a statement.

    Reads the statement's group text (via ``statement_of``) and writes it to
    each attached procedure's ``content``. For now the full group text is used;
    an LLM pass will extract just the procedure portion.

    Args:
        statements: The statement overlay.
        nodes_by_id: The full node stream keyed by stable id. It holds the
            groups' members, never the statements themselves.
    """
    for statement in statements:
        member_ids = statement.statement_of or []
        members = [
            nodes_by_id[node_id]
            for node_id in member_ids
            if node_id in nodes_by_id
        ]
        text = statement_extractor.group_text(members)
        for procedure in statement.procedures:
            procedure.content = text or None

    logger.info(
        'procedure extractor: %d procedure(s) from %d statement(s)',
        sum(len(statement.procedures) for statement in statements),
        len(statements),
    )


class ProcedureExtractorNode:
    """Fills ``content`` on each Procedure attached to a statement."""

    async def run(self, state: state.State) -> dict:
        """Fill procedure content from group text.

        Args:
            state: The pipeline state, holding the node stream and the
                statement overlay.

        Returns:
            An empty update — the procedures are mutated in place.
        """
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        statements = state.get('statements', [])
        if statements:
            extract_procedures(statements, nodes_by_id)
        return {}
