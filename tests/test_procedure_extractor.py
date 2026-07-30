"""Procedure extractor: fills content on procedures attached to statements."""

import asyncio

from kms.core import models
from kms.ingestion import procedure_extractor


def test_extract_fills_procedure_content_from_group_text():
    # The statement is beside the stream; the stream holds its members.
    stmt = models.StatementNode(content='Theorem.', id=0, statement_of=[0, 1])
    stmt.procedures.append(models.Procedure(index=0))
    nodes = [
        models.ParagraphNode(content='Theorem.', id=0),
        models.ParagraphNode(content='Proof. done.', id=1),
    ]
    by_id = {n.id: n for n in nodes}
    procedure_extractor.extract_procedures([stmt], by_id)
    assert stmt.procedures[0].content == 'Theorem.\n\nProof. done.'


def test_procedures_on_one_statement():
    stmt = models.StatementNode(
        content='Theorem.', id=0, statement_of=[0, 1, 2]
    )
    stmt.procedures.append(models.Procedure(index=0))
    stmt.procedures.append(models.Procedure(index=1))
    nodes = [
        models.ParagraphNode(content='Theorem.', id=0),
        models.ParagraphNode(content='Proof 1.', id=1),
        models.ParagraphNode(content='Proof 2.', id=2),
    ]
    by_id = {n.id: n for n in nodes}
    procedure_extractor.extract_procedures([stmt], by_id)
    assert stmt.procedures[0].content is not None
    assert stmt.procedures[1].content is not None


def test_node_run_fills_procedure_content():
    stmt = models.StatementNode(content='Theorem.', id=0, statement_of=[0, 1])
    stmt.procedures.append(models.Procedure(index=0))
    nodes = [
        models.ParagraphNode(content='Theorem.', id=0),
        models.ParagraphNode(content='Proof.', id=1),
    ]
    node = procedure_extractor.ProcedureExtractorNode()
    asyncio.run(node.run({'nodes': nodes, 'statements': [stmt]}))
    assert stmt.procedures[0].content == 'Theorem.\n\nProof.'


def test_node_run_without_procedures_is_noop():
    stmt = models.StatementNode(content='x', id=0, statement_of=[0])
    nodes = [models.ParagraphNode(content='x', id=0)]
    node = procedure_extractor.ProcedureExtractorNode()
    asyncio.run(node.run({'nodes': nodes, 'statements': [stmt]}))
    assert stmt.procedures == []
