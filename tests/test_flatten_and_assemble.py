"""Flat-stream refactor: seam-birthed global node list + assembler resolving by segment_index."""

import pathlib
import tempfile

from kms.core import models
from kms.output.assembler import assemble


def _segments():
    return [
        models.Segment(
            index=0,
            image_path='p0.png',
            pictures=[],
            nodes=[
                models.ASTNode(type=models.NodeType.HEADER, content='# Ch 1'),
                models.ASTNode(type=models.NodeType.PARAGRAPH, content='intro'),
            ],
        ),
        models.Segment(
            index=1,
            image_path='p1.png',
            pictures=[],
            nodes=[
                models.ASTNode(
                    type=models.NodeType.PARAGRAPH, content='body ![1]() fig'
                ),
                models.ASTNode(
                    type=models.NodeType.PARAGRAPH, content='1. solve x'
                ),
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
    out = assemble(flat, segs, output_dir=tempfile.mkdtemp(), filename='doc.md')
    text = pathlib.Path(out).read_text()
    assert '# Ch 1' in text and '1. solve x' in text
    assert (
        '![1]()' in text
    )  # no matching picture -> placeholder passes through, no crash
