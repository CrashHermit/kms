import asyncio

from kms2.core.model import Source, SourceBlock, SourcePage, VisualAsset
from kms2.database.source.queries import REPLACE_SOURCE
from kms2.database.source.repository import SourceRepository


class _RecordingResult:
    async def consume(self) -> None:
        return None


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run(self, query: str, **parameters: object) -> _RecordingResult:
        self.calls.append((query, parameters))
        return _RecordingResult()


class _SessionContext:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> _RecordingSession:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_replace_source_projects_ordered_structure_once():
    first_asset = VisualAsset(uuid='asset-1', path='first.png')
    second_asset = VisualAsset(uuid='asset-2', path='second.png')
    first_block = SourceBlock(
        uuid='block-1',
        block_type='text',
        content='first block',
        crop_path='first-crop.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[first_asset, second_asset],
    )
    second_block = SourceBlock(
        uuid='block-2',
        block_type='image',
        content=None,
    )
    pages = [
        SourcePage(index=0),
        SourcePage(index=1, blocks=[first_block, second_block]),
    ]
    session = _RecordingSession()

    asyncio.run(
        SourceRepository(lambda: _SessionContext(session)).replace_source(
            Source(uuid='source-1', key='book.pdf', metadata={'kind': 'book'}),
            pages,
        )
    )

    assert len(session.calls) == 1
    query, parameters = session.calls[0]
    assert query is REPLACE_SOURCE
    assert parameters == {
        'source': {
            'uuid': 'source-1',
            'key': 'book.pdf',
            'metadata': '{"kind": "book"}',
        },
        'pages': [{'index': 0}, {'index': 1}],
        'blocks': [
            {
                'uuid': 'block-1',
                'block_type': 'text',
                'content': 'first block',
                'crop_path': 'first-crop.png',
                'crop_bbox': [1, 2, 3, 4],
            },
            {
                'uuid': 'block-2',
                'block_type': 'image',
                'content': None,
                'crop_path': None,
                'crop_bbox': None,
            },
        ],
        'assets': [
            {'uuid': 'asset-1', 'path': 'first.png'},
            {'uuid': 'asset-2', 'path': 'second.png'},
        ],
        'block_asset_pairs': [
            {'block_uuid': 'block-1', 'asset_uuid': 'asset-1'},
            {'block_uuid': 'block-1', 'asset_uuid': 'asset-2'},
        ],
        'asset_pairs': [{'from_uuid': 'asset-1', 'to_uuid': 'asset-2'}],
        'asset_bounds': [
            {
                'block_uuid': 'block-1',
                'first_asset_uuid': 'asset-1',
                'last_asset_uuid': 'asset-2',
            }
        ],
        'page_block_pairs': [
            {'page_index': 1, 'block_uuid': 'block-1'},
            {'page_index': 1, 'block_uuid': 'block-2'},
        ],
        'page_pairs': [{'from_index': 0, 'to_index': 1}],
        'block_pairs': [{'from_uuid': 'block-1', 'to_uuid': 'block-2'}],
        'page_bounds': [
            {
                'page_index': 1,
                'first_block_uuid': 'block-1',
                'last_block_uuid': 'block-2',
            }
        ],
        'first_page_index': 0,
        'first_block_uuid': 'block-1',
        'last_block_uuid': 'block-2',
    }
    assert 'MATCH (source)-[:HAS_PAGE]->(from_page:SourcePage' in query
    assert 'MATCH (source)-[:HAS_PAGE]->(to_page:SourcePage' in query
    for block_type in (
        'text',
        'equation',
        'paragraph',
        'math',
        'code',
        'list',
        'table',
        'image',
        'caption',
        'header',
        'bibliographic',
        'note',
        'footer',
        'aside_text',
        'markdown',
        'instruction',
    ):
        assert f"row.block_type = '{block_type}'" in query
