"""The collector fan-in: whichever finder overlays ran become one document-ordered, globally-id'd
`entities` list. Pure logic, no LLM and no database."""

import asyncio

from kms.core import models
from kms.entity.collector import CollectorNode


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content=str(i),
            id=i,
            segment_index=0,
        )
        for i in range(6)
    ]


def _run(state):
    return asyncio.run(CollectorNode().run(state))


def test_collects_the_three_per_type_channels_in_document_order():
    out = _run(
        {
            'nodes': _nodes(),
            'problem_entities': [models.Entity(type='problem', members=[3])],
            'definition_entities': [
                models.Entity(type='definition', members=[0])
            ],
            'theorem_entities': [models.Entity(type='theorem', members=[1])],
        }
    )
    assert list(out) == ['entities']
    assert [(e.id, e.type) for e in out['entities']] == [
        (0, 'definition'),
        (1, 'theorem'),
        (2, 'problem'),
    ]


def test_collects_the_block_channel_alone():
    # the block layer writes one channel; the collector's contract does not change
    out = _run(
        {
            'nodes': _nodes(),
            'block_entities': [
                models.Entity(type='law', members=[2]),
                models.Entity(members=[0]),
            ],
        }
    )
    assert [(e.id, e.type) for e in out['entities']] == [(0, None), (1, 'law')]


def test_absent_channels_contribute_nothing():
    assert _run({'nodes': _nodes()}) == {'entities': []}
