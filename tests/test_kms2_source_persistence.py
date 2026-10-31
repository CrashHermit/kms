import asyncio

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_pedagogical.persistence import (
    SourcePedagogicalPersistenceNode,
)
from kms2.node.source_processing.persistence import SourcePersistenceNode


class _RecordingRepository:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def replace_source(
        self,
        source: Source,
        pages: list[SourcePage],
    ) -> None:
        self._events.append(('source', source, pages))

    async def replace_pedagogical_material(
        self,
        source_uuid: str,
        pages: list[SourcePage],
        instructions: list[Instruction],
        statements: list[StatementDraft],
        procedures: list[ProcedureDraft],
    ) -> None:
        self._events.append(
            (
                'pedagogical',
                source_uuid,
                pages,
                instructions,
                statements,
                procedures,
            )
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

    result = asyncio.run(
        SourcePersistenceNode(
            _RecordingRepository(events),
        ).run(
            SourceProcessingState(
                pdf_path='book.pdf',
                source=source,
                embedded_pages=pages,
            )
        )
    )

    assert events == [('source', source, pages)]
    assert result == {}


def test_pedagogical_persistence_replaces_only_stage_material():
    events: list[object] = []
    pages = [SourcePage(index=0, blocks=[])]
    instructions = [Instruction(uuid='instruction-1')]
    statements = [StatementDraft(uuid='statement-1', is_exercise=True)]
    procedures = [ProcedureDraft(uuid='procedure-1')]

    result = asyncio.run(
        SourcePedagogicalPersistenceNode(
            _RecordingRepository(events),
        ).run(
            SourcePedagogicalState(
                source_uuid='source-1',
                embedded_pages=pages,
                instructions=instructions,
                statements=statements,
                procedures=procedures,
            )
        )
    )

    assert events == [
        (
            'pedagogical',
            'source-1',
            pages,
            instructions,
            statements,
            procedures,
        )
    ]
    assert result == {}
