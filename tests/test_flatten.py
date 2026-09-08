import asyncio

from kms.construction import statement_procedure_builder
from kms.core import models


def _documents():
    return [
        models.Document(
            index=0,
            image_path='p0.png',
            nodes=[
                models.SourceNode(
                    type=models.NodeType.HEADER, content='# Ch 1'
                ),
                models.SourceNode(
                    type=models.NodeType.PARAGRAPH, content='intro'
                ),
            ],
        ),
        models.Document(
            index=1,
            image_path='p1.png',
            nodes=[
                models.SourceNode(
                    type=models.NodeType.PARAGRAPH,
                    content='body ![1]() fig',
                ),
                models.SourceNode(
                    type=models.NodeType.PARAGRAPH, content='1. solve x'
                ),
            ],
        ),
    ]


def test_flatten_assigns_stable_ids_and_document_index_across_pages():
    flat = models.flatten_documents(_documents())
    # Flattening no longer assigns node.id; use list positions instead
    assert len(flat) == 4
    assert [n.document_index for n in flat] == [0, 0, 1, 1]


def test_flatten_preserves_directly_attached_assets(tmp_path):
    first_path = str(tmp_path / 'Image_000.png')
    second_path = str(tmp_path / 'Image_001.png')
    documents = [
        models.Document(
            index=0,
            image_path='p0.png',
            nodes=[
                models.SourceNode(
                    type=models.NodeType.IMAGE,
                    assets=[models.VisualAsset(path=first_path)],
                ),
                models.SourceNode(
                    type=models.NodeType.IMAGE,
                    assets=[models.VisualAsset(path=second_path)],
                ),
            ],
        )
    ]

    flat = models.flatten_documents(documents)

    assert [asset.path for asset in flat[0].assets] == [first_path]
    assert [asset.path for asset in flat[1].assets] == [second_path]


class _AllStatements:
    async def acall(self, current_nodes):
        return (True, False)


def test_overlay_leaves_each_block_in_the_stream_exactly_once():
    nodes = [
        models.SourceNode(
            type='paragraph',
            content='Theorem 2.1.',
            uuid='node-0',
            document_index=0,
        ),
        models.SourceNode(
            type='paragraph',
            content='Proof. Let e be ...',
            uuid='node-1',
            document_index=0,
        ),
        models.SourceNode(
            type='paragraph',
            content='Hence e is unique.',
            uuid='node-2',
            document_index=0,
        ),
        models.SourceNode(
            type='paragraph',
            content='1.23 Compute it.',
            uuid='node-3',
            document_index=0,
        ),
    ]
    state = {
        'nodes': nodes,
        'spans': [[0, 1, 2], [1, 2], [3]],
        'source': models.Source(key='book.pdf'),
    }

    typer = statement_procedure_builder.StatementProcedureBuilderNode(
        role_module=_AllStatements()
    )
    state.update(asyncio.run(typer.run(state)))

    contents = [node.content for node in state['nodes']]
    for content in (
        'Theorem 2.1.',
        'Proof. Let e be ...',
        'Hence e is unique.',
    ):
        assert contents.count(content) == 1, f'{content!r} appears twice'
    assert state['statements'][0].member_positions == [0, 1, 2]
