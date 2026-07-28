"""Statement extractor: concatenates group member text into ``content``."""

import asyncio

from kms.core import models
from kms.ingestion import statement_extractor


def test_group_text_joins_and_drops_blank_nodes():
    nodes = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Every group has an identity.', id=1),
        models.ParagraphNode(content='  ', id=2),
    ]
    assert statement_extractor.group_text(nodes) == (
        'Theorem 2.1\n\nEvery group has an identity.'
    )


def test_extract_fills_content_from_statement_of():
    stmt = models.StatementNode(
        content='Theorem 2.1', id=0, statement_of=[0, 1, 2]
    )
    members = [
        models.ParagraphNode(content='Theorem 2.1', id=0),
        models.ParagraphNode(content='Every group has an identity.', id=1),
        models.ParagraphNode(content='  ', id=2),
    ]
    statement_extractor.extract_statement(
        stmt, {n.id: n for n in members}
    )
    assert stmt.content == 'Theorem 2.1\n\nEvery group has an identity.'


def test_extract_skips_missing_member_ids():
    stmt = models.StatementNode(
        content='Thm', id=0, statement_of=[0, 1, 99]
    )
    members = [
        models.ParagraphNode(content='a', id=0),
        models.ParagraphNode(content='b', id=1),
    ]
    statement_extractor.extract_statement(
        stmt, {n.id: n for n in members}
    )
    assert stmt.content == 'a\n\nb'


def test_node_run_fills_content():
    stmt = models.StatementNode(
        content='Thm', id=0, statement_of=[1, 2]
    )
    members = [
        models.ParagraphNode(content='Theorem 2.1', id=1),
        models.ParagraphNode(content='Every group has an identity.', id=2),
    ]
    node = statement_extractor.StatementExtractorNode()
    asyncio.run(
        node.run({'nodes': [stmt] + members, 'statement_ids': [0]})
    )
    assert stmt.content == 'Theorem 2.1\n\nEvery group has an identity.'


def test_node_run_on_empty_is_noop():
    node = statement_extractor.StatementExtractorNode()
    assert asyncio.run(
        node.run({'nodes': [], 'statement_ids': []})
    ) == {}
