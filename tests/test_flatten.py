"""Flat-stream refactor: the seam-birthed global node list."""

import asyncio

from kms.core import models
from kms.ingestion import hub_builder


def _segments():
    return [
        models.Segment(
            index=0,
            image_path='p0.png',
            pictures=[],
            nodes=[
                models.ASTNode(type='header', content='# Ch 1'),
                models.ASTNode(type='paragraph', content='intro'),
            ],
        ),
        models.Segment(
            index=1,
            image_path='p1.png',
            pictures=[],
            nodes=[
                models.ASTNode(type='paragraph', content='body ![1]() fig'),
                models.ASTNode(type='paragraph', content='1. solve x'),
            ],
        ),
    ]


def test_flatten_assigns_stable_ids_and_seg_index_across_pages():
    flat = models.flatten_segments(_segments())
    assert [n.id for n in flat] == [0, 1, 2, 3]
    assert [n.segment_index for n in flat] == [
        0,
        0,
        1,
        1,
    ]  # provenance survives the flatten


class _AllStatements:
    """Stands in for the LLM: every span is a statement."""

    async def acall(self, contents):
        return (True, False)


def test_overlay_leaves_each_block_in_the_stream_exactly_once():
    # Regression: the role typer used to swap each span's first node for its
    # Statement inside `nodes`, and the statement extractor then set that
    # node's content to the WHOLE group's text — so every member after the
    # first was represented twice, once inside the fused statement and once as
    # itself, and the persister wrote it that way. The overlay now travels on
    # its own channel and carries no text.
    nodes = [
        models.ASTNode(type='paragraph', content='Theorem 2.1.', id=0, segment_index=0),
        models.ASTNode(type='paragraph', 
            content='Proof. Let e be ...', id=1, segment_index=0
        ),
        models.ASTNode(type='paragraph', 
            content='Hence e is unique.', id=2, segment_index=0
        ),
        models.ASTNode(type='paragraph', content='1.23 Compute it.', id=3, segment_index=0),
    ]
    # One multi-node group and one single-node group, plus an overlapping span
    # (the component finder is allowed to emit those).
    state = {'nodes': nodes, 'spans': [[0, 1, 2], [1, 2], [3]]}

    typer = hub_builder.HubBuilderNode(role_module=_AllStatements())
    state.update(asyncio.run(typer.run(state)))

    contents = [node.content for node in state['nodes']]
    for content in (
        'Theorem 2.1.',
        'Proof. Let e be ...',
        'Hence e is unique.',
    ):
        assert contents.count(content) == 1, f'{content!r} appears twice'
    # The overlay carries hub membership, not text.
    assert state['statements'][0].members == [0, 1, 2]
