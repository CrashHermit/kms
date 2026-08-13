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
    ]


def test_flatten_resolves_image_nodes_to_picture_paths(tmp_path):
    first_path = str(tmp_path / 'Image_000.png')
    second_path = str(tmp_path / 'Image_001.png')
    segments = [
        models.Segment(
            index=0,
            image_path='p0.png',
            pictures=[
                models.Picture(index=1, image_path=first_path),
                models.Picture(index=2, image_path=second_path),
            ],
            nodes=[
                models.ASTNode(type='paragraph', content='see figure'),
                models.ASTNode(type='image'),
                models.ASTNode(type='image', content='![2]()'),
            ],
        )
    ]
    flat = models.flatten_segments(segments)
    assert flat[0].image_path is None
    assert flat[1].image_path == first_path
    assert flat[2].image_path == second_path


def test_flatten_ignores_pictures_without_image_nodes(tmp_path):
    used_path = str(tmp_path / 'Image_000.png')
    segments = [
        models.Segment(
            index=0,
            image_path='p0.png',
            pictures=[
                models.Picture(index=1, image_path=used_path),
                models.Picture(
                    index=2, image_path=str(tmp_path / 'orphan.png')
                ),
            ],
            nodes=[models.ASTNode(type='image')],
        )
    ]
    flat = models.flatten_segments(segments)
    assert flat[0].image_path == used_path


class _AllStatements:
    async def acall(self, contents):
        return (True, False)


def test_overlay_leaves_each_block_in_the_stream_exactly_once():
    nodes = [
        models.ASTNode(
            type='paragraph', content='Theorem 2.1.', id=0, segment_index=0
        ),
        models.ASTNode(
            type='paragraph',
            content='Proof. Let e be ...',
            id=1,
            segment_index=0,
        ),
        models.ASTNode(
            type='paragraph',
            content='Hence e is unique.',
            id=2,
            segment_index=0,
        ),
        models.ASTNode(
            type='paragraph', content='1.23 Compute it.', id=3, segment_index=0
        ),
    ]
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
    assert state['statements'][0].members == [0, 1, 2]
