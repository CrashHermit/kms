"""Statement extractor: the attribute transcription pass over a typed block. Covers
label/number/title, the deterministic contents assembly (label peeling), and the graph-node
wrapper. The open `type` is `block_typer`'s job and is tested there. The LLM call is stubbed —
this tests everything around it."""

import asyncio

from kms.core import models
from kms.entity import statement_extractor


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.HEADER,
            content='Theorem 2.1',
            id=0,
            segment_index=0,
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Every group has an identity.',
            id=1,
            segment_index=0,
        ),
    ]


class _ScriptedModule:
    """A stand-in Module returning a fixed Identity, recording what it was asked about."""

    def __init__(self, identity):
        self.identity_result = identity
        self.seen: list[list[models.ASTNode]] = []

    async def acall(self, members):
        self.seen.append(members)
        return self.identity_result


def _identity(**kwargs):
    return statement_extractor.Identity(**kwargs)


# --- contents assembly (deterministic, no LLM) ---


def test_contents_drops_a_standalone_label_node():
    nodes = _nodes()
    assert statement_extractor.contents_of(nodes, 'Theorem 2.1') == [
        'Every group has an identity.'
    ]


def test_contents_peels_a_fused_label_but_keeps_the_statement():
    fused = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Theorem 2.1. Every group has an identity.',
            id=0,
        )
    ]
    assert statement_extractor.contents_of(fused, 'Theorem 2.1') == [
        'Every group has an identity.'
    ]


def test_contents_peels_nothing_without_a_label():
    nodes = _nodes()
    assert statement_extractor.contents_of(nodes) == [
        'Theorem 2.1',
        'Every group has an identity.',
    ]


def test_contents_leaves_text_that_does_not_start_with_the_label():
    nodes = _nodes()
    assert (
        statement_extractor.contents_of(nodes, 'Example 9')[0] == 'Theorem 2.1'
    )


# --- the attribute pass ---


def test_extract_fills_label_number_title_and_contents():
    nodes = _nodes()
    entity = models.Entity(members=[0, 1])
    module = _ScriptedModule(
        _identity(
            label='Theorem 2.1',
            number='2.1',
            title='Group Identity',
        )
    )
    asyncio.run(
        statement_extractor.extract_statement(
            entity, {node.id: node for node in nodes}, module
        )
    )
    assert entity.label == 'Theorem 2.1'
    assert entity.number == '2.1' and entity.title == 'Group Identity'
    assert entity.contents == ['Every group has an identity.']


def test_extract_never_restructures_the_span():
    # the finder decided the extent; this pass only reads it (no proof_start/solution_start)
    nodes = _nodes()
    entity = models.Entity(members=[0, 1])
    module = _ScriptedModule(_identity(label='Theorem 2.1'))
    asyncio.run(
        statement_extractor.extract_statement(
            entity, {node.id: node for node in nodes}, module
        )
    )
    assert entity.members == [0, 1]
    assert entity.procedures == []


def test_extract_skips_member_ids_missing_from_the_stream():
    nodes = _nodes()
    entity = models.Entity(members=[0, 1, 99])
    module = _ScriptedModule(_identity(label='Theorem 2.1'))
    asyncio.run(
        statement_extractor.extract_statement(
            entity, {node.id: node for node in nodes}, module
        )
    )
    assert [node.id for node in module.seen[0]] == [0, 1]


# --- graph node ---


def test_node_run_writes_the_entities_channel():
    nodes = _nodes()
    entity = models.Entity(members=[0, 1])
    node = statement_extractor.StatementExtractorNode(
        module=_ScriptedModule(_identity(label='Definition 2.1', number='2.1'))
    )
    out = asyncio.run(node.run({'nodes': nodes, 'entities': [entity]}))
    assert list(out) == ['entities']
    assert out['entities'][0].number == '2.1'


def test_node_run_on_an_empty_overlay_is_a_noop():
    node = statement_extractor.StatementExtractorNode(
        module=_ScriptedModule(_identity())
    )
    assert asyncio.run(node.run({'nodes': [], 'entities': []})) == {
        'entities': []
    }
