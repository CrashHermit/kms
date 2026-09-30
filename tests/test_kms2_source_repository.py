import asyncio

import pytest

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.instruction import Instruction
from kms2.core.model.source_processing.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
    SourceOwnershipConflict,
)
from kms2.database.source.source_graph_repository import SourceGraphRepository


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
        StatementDraft(
            uuid='statement-1',
            member_block_uuids=['block-2'],
            is_exercise=True,
        )
    ]
    procedures = [
        ProcedureDraft(uuid='procedure-1', member_block_uuids=['block-1'])
    ]
    session = _RecordingSession()

    asyncio.run(
        SourceGraphRepository(lambda: _SessionContext(session)).replace_source(
            Source(uuid='source-1', key='book.pdf', metadata={'kind': 'book'}),
            pages,
            instructions,
            statements,
            procedures,
        )
    )

    parameters = session.calls[0][1]
    assert parameters['statements'] == [
        {
            'uuid': 'statement-1',
            'source_uuid': 'source-1',
            'is_exercise': True,
        }
    ]
    assert parameters['procedures'] == [
        {'uuid': 'procedure-1', 'source_uuid': 'source-1'}
    ]
    assert parameters['statement_member_pairs'] == [
        {'block_uuid': 'block-2', 'statement_uuid': 'statement-1'}
    ]
    assert parameters['procedure_member_pairs'] == [
        {'block_uuid': 'block-1', 'procedure_uuid': 'procedure-1'}
    ]
    assert parameters['instruction_governance_pairs'] == [
        {'instruction_uuid': 'instruction-1', 'statement_uuid': 'statement-1'}
    ]


def test_list_sources_returns_stable_user_source_summaries():
    session = _RecordingSession(
        _RecordingResult(
            [
                {'uuid': 'source-2', 'key': 'second.pdf'},
                {'uuid': 'source-1', 'key': 'first.pdf'},
            ]
        )
    )

    sources = asyncio.run(
        SourceCatalogRepository(lambda: _SessionContext(session)).list_sources(
            'user-1'
        )
    )

    assert sources == [
        Source(uuid='source-2', key='second.pdf'),
        Source(uuid='source-1', key='first.pdf'),
    ]


def test_list_unowned_sources_excludes_owned_source_catalog():
    session = _RecordingSession(
        _RecordingResult([{'uuid': 'legacy-1', 'key': 'legacy.pdf'}])
    )

    sources = asyncio.run(
        SourceCatalogRepository(
            lambda: _SessionContext(session)
        ).list_unowned_sources()
    )

    assert sources == [Source(uuid='legacy-1', key='legacy.pdf')]


def test_adopt_source_locks_source_and_rejects_claimed_source():
    session = _RecordingSession()
    repository = SourceCatalogRepository(lambda: _SessionContext(session))

    with pytest.raises(SourceOwnershipConflict):
        asyncio.run(repository.adopt_source('user-1', 'source-1'))


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
    repository = SourceBlockRepository(lambda: _SessionContext(session))

    results = asyncio.run(
        repository.find_similar_blocks('query-block', top_k=2)
    )

    assert [result.uuid for result in results] == [
        'nearest-block',
        'next-block',
    ]
    assert results[0].score == 0.99
    assert 'embedding' not in results[0].model_dump()
