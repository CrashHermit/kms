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
    statement_ids: list[int],
    nodes_by_id: dict[int, models.ASTNode],
) -> None:
    """Fill ``content`` on every Procedure attached to a statement.

    Reads the statement's group text (via ``statement_of``) and writes it to
    each attached procedure's ``content``. For now the full group text is used;
    an LLM pass will extract just the procedure portion.

    Args:
        statement_ids: The first-node ids of every group.
        nodes_by_id: The full node stream keyed by stable id.
    """
    for statement_id in statement_ids:
        statement = nodes_by_id.get(statement_id)
        if not isinstance(statement, models.StatementNode):
            continue
        member_ids = statement.statement_of or []
        members = [
            nodes_by_id[node_id]
            for node_id in member_ids
            if node_id in nodes_by_id
        ]
        text = statement_extractor.group_text(members)
        for procedure in statement.procedures:
            procedure.content = text or None

    procedure_count = sum(
        len(nodes_by_id[statement_id].procedures)
        for statement_id in statement_ids
        if statement_id in nodes_by_id
        and isinstance(nodes_by_id[statement_id], models.StatementNode)
    )
    logger.info(
        'procedure extractor: %d procedure(s) from %d statement(s)',
        procedure_count,
        len(statement_ids),
    )


class ProcedureExtractorNode:
    """Fills ``content`` on each Procedure attached to a statement."""

    async def run(self, state: state.State) -> dict:
        """Fill procedure content from group text.

        Args:
            state: The pipeline state, holding the nodes and statement ids.

        Returns:
            An empty update — the procedures are mutated in place.
        """
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        statement_ids = state.get('statement_ids', [])
        if statement_ids:
            extract_procedures(statement_ids, nodes_by_id)
        return {}
