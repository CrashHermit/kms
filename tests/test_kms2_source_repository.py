import asyncio

from kms2.core.model import (
    Instruction,
    Procedure,
    Source,
    SourceBlock,
    SourcePage,
    Statement,
    VisualAsset,
)
from kms2.database.source.queries import (
    FIND_SIMILAR_SOURCE_BLOCKS,
    READ_SOURCES,
    REPLACE_SOURCE,
)
from kms2.database.source.repository import SourceRepository


class _RecordingResult:
    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        self._rows = rows or []

    async def consume(self) -> None:
        return None

    async def data(self) -> list[dict[str, str]]:
        return self._rows


class _RecordingSession:
    def __init__(self, result: _RecordingResult | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._result = result or _RecordingResult()

    async def run(self, query: str, **parameters: object) -> _RecordingResult:
        self.calls.append((query, parameters))
        return self._result


class _SessionContext:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> _RecordingSession:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_replace_source_projects_pointer_rows_and_governance_once():
    first_asset = VisualAsset(uuid='asset-1', path='first.png')
    first_block = SourceBlock(
        uuid='block-1',
        block_type='paragraph',
        content='first block',
        assets=[first_asset],
    )
    second_block = SourceBlock(
        uuid='block-2',
        block_type='image',
        content=None,
    )
    pages = [SourcePage(index=1, blocks=[first_block, second_block])]
    instructions = [
        Instruction(
            uuid='instruction-1',
            member_block_uuids=['block-1'],
            governed_statement_uuids=['statement-1'],
        )
    ]
    statements = [
        Statement(
            uuid='statement-1',
            member_block_uuids=['block-2'],
            is_exercise=True,
        )
    ]
    procedures = [Procedure(uuid='procedure-1', member_block_uuids=['block-1'])]
    session = _RecordingSession()

    asyncio.run(
        SourceRepository(lambda: _SessionContext(session)).replace_source(
            Source(uuid='source-1', key='book.pdf', metadata={'kind': 'book'}),
            pages,
            instructions,
            statements,
            procedures,
        )
    )

    assert len(session.calls) == 1
    query, parameters = session.calls[0]
    assert query is REPLACE_SOURCE
    assert parameters['statements'] == [
        {'uuid': 'statement-1', 'is_exercise': True}
    ]
    assert parameters['procedures'] == [{'uuid': 'procedure-1'}]
    assert parameters['statement_member_pairs'] == [
        {'block_uuid': 'block-2', 'statement_uuid': 'statement-1'}
    ]
    assert parameters['procedure_member_pairs'] == [
        {'block_uuid': 'block-1', 'procedure_uuid': 'procedure-1'}
    ]
    assert parameters['instruction_governance_pairs'] == [
        {'instruction_uuid': 'instruction-1', 'statement_uuid': 'statement-1'}
    ]
    assert 'HAS_STATEMENT' not in query
    assert 'HAS_PROCEDURE' not in query
    assert 'GOVERNS]->(block' not in query
    assert 'MEMBER_OF]->(statement' in query
    assert 'MEMBER_OF]->(procedure' in query


def test_list_sources_returns_stable_source_summaries():
    session = _RecordingSession(
        _RecordingResult(
            [
                {'uuid': 'source-2', 'key': 'second.pdf'},
                {'uuid': 'source-1', 'key': 'first.pdf'},
            ]
        )
    )

    sources = asyncio.run(
        SourceRepository(lambda: _SessionContext(session)).list_sources()
    )

    assert sources == [
        Source(uuid='source-2', key='second.pdf'),
        Source(uuid='source-1', key='first.pdf'),
    ]
    assert session.calls == [(READ_SOURCES, {})]


def test_find_similar_blocks_uses_neo4j_vector_index_and_omits_embedding():
    session = _RecordingSession(
        _RecordingResult(
            [
                {
                    'uuid': 'nearest-block',
                    'block_type': 'paragraph',
                    'content': 'nearest',
                    'score': 0.99,
                },
                {
                    'uuid': 'next-block',
                    'block_type': 'equation',
                    'content': None,
                    'score': 0.81,
                },
            ]
        )
    )
    repository = SourceRepository(lambda: _SessionContext(session))

    results = asyncio.run(
        repository.find_similar_blocks('query-block', top_k=2)
    )

    assert [result.uuid for result in results] == [
        'nearest-block',
        'next-block',
    ]
    assert results[0].score == 0.99
    assert 'embedding' not in results[0].model_dump()
    assert session.calls == [
        (
            FIND_SIMILAR_SOURCE_BLOCKS,
            {
                'query_uuid': 'query-block',
                'top_k': 2,
                'candidate_limit': 3,
            },
        )
    ]
    assert 'source_block_embedding' in FIND_SIMILAR_SOURCE_BLOCKS
    assert 'node.uuid <> query.uuid' in FIND_SIMILAR_SOURCE_BLOCKS
    assert 'ORDER BY score DESC, uuid ASC' in FIND_SIMILAR_SOURCE_BLOCKS
