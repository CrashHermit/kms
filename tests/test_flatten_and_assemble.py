"""Flat-stream refactor: seam-birthed global node list + assembler resolving by
segment_index."""

import asyncio
import tempfile

from kms.core import models
from kms.ingestion import role_typer
from kms.output import assembler


def _segments():
    return [
        models.Segment(
            index=0,
            image_path='p0.png',
            pictures=[],
            nodes=[
                models.HeaderNode(content='# Ch 1'),
                models.ParagraphNode(content='intro'),
            ],
        ),
        models.Segment(
            index=1,
            image_path='p1.png',
            pictures=[],
            nodes=[
                models.ParagraphNode(content='body ![1]() fig'),
                models.ParagraphNode(content='1. solve x'),
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


def test_assemble_walks_flat_nodes_and_passes_unmatched_placeholder():
    segs = _segments()
    flat = models.flatten_segments(segs)
    text = assembler.assemble(flat, segs, output_dir=tempfile.mkdtemp())
    assert '# Ch 1' in text and '1. solve x' in text
    assert (
        '![1]()' in text
    )  # no matching picture -> placeholder passes through, no crash


class _AllStatements:
    """Stands in for the LLM: every span is a statement."""

    async def acall(self, contents):
        return [role_typer.STATEMENT_ROLE]


def test_assembly_emits_each_block_once_after_the_overlay_is_built():
    # Regression: the role typer used to swap each span's first node for its
    # Statement inside `nodes`, and the statement extractor then set that
    # node's content to the WHOLE group's text — so the assembler emitted every
    # member after the first twice, once inside the fused statement and once as
    # itself. The overlay now travels on its own channel and carries no text.
    nodes = [
        models.ParagraphNode(content='Theorem 2.1.', id=0, segment_index=0),
        models.ParagraphNode(
            content='Proof. Let e be ...', id=1, segment_index=0
        ),
        models.ParagraphNode(
            content='Hence e is unique.', id=2, segment_index=0
        ),
        models.ParagraphNode(content='1.23 Compute it.', id=3, segment_index=0),
    ]
    # One multi-node group and one single-node group, plus an overlapping span
    # (the component finder is allowed to emit those).
    state = {'nodes': nodes, 'spans': [[0, 1, 2], [1, 2], [3]]}

    typer = role_typer.RoleTyperNode(module=_AllStatements())
    state.update(asyncio.run(typer.run(state)))

    text = assembler.assemble(
        state['nodes'],
        [models.Segment(index=0, image_path='p0.png')],
        output_dir=tempfile.mkdtemp(),
    )
    for content in (
        'Theorem 2.1.',
        'Proof. Let e be ...',
        'Hence e is unique.',
    ):
        assert text.count(content) == 1, f'{content!r} appears twice'
    # The overlay carries hub membership, not text.
    assert state['statements'][0].members == [0, 1, 2]
