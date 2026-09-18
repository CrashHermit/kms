import asyncio

from kms2.core.model import (
    Instruction,
    ProcedureDraft,
    Source,
    SourceBlock,
    SourcePage,
    StatementDraft,
)
from kms2.langgraph.source.state import SourceState
from kms2.node.source.persistence import SourcePersistenceNode


class _RecordingRepository:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def replace_source(
        self,
        source: Source,
        pages: list[SourcePage],
        instructions: list[Instruction],
        statements: list[StatementDraft],
        procedures: list[ProcedureDraft],
    ) -> None:
        self._events.append(
            ('repository', source, pages, instructions, statements, procedures)
        )


def test_persistence_replaces_source_without_schema_side_effects():
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
    statements = [StatementDraft(uuid='statement-1', is_exercise=True)]
    procedures = [ProcedureDraft(uuid='procedure-1')]

    result = asyncio.run(
        SourcePersistenceNode(
            _RecordingRepository(events),
        ).run(
            SourceState(
                pdf_path='book.pdf',
                source=source,
                embedded_pages=pages,
                instructions=[Instruction(uuid='instruction-1')],
                statements=statements,
                procedures=procedures,
            )
        )
    )

    assert events == [
        (
            'repository',
            source,
            pages,
            [Instruction(uuid='instruction-1')],
            statements,
            procedures,
        )
    ]
    assert result == {}
