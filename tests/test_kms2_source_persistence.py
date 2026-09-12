import asyncio

from kms2.core.model import Source, SourceBlock, SourcePage
from kms2.langgraph.source.state import SourceState
from kms2.node.source.persistence import SourcePersistenceNode


class _RecordingRepository:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def replace_source(
        self,
        source: Source,
        pages: list[SourcePage],
    ) -> None:
        self._events.append(('repository', source, pages))


def test_persistence_initializes_schema_before_replacing_source():
    events: list[object] = []
    source = Source(uuid='source-1', key='book.pdf')
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    uuid='block-1',
                    block_type='paragraph',
                    embedding=[0.1, 0.2],
                )
            ],
        )
    ]

    async def initialize_schema() -> None:
        events.append('schema')

    result = asyncio.run(
        SourcePersistenceNode(
            _RecordingRepository(events),
            initialize_schema,
        ).run(
            SourceState(
                pdf_path='book.pdf',
                source=source,
                embedded_pages=pages,
            )
        )
    )

    assert events == ['schema', ('repository', source, pages)]
    assert result == {}
