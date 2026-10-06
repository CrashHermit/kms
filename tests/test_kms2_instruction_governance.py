import asyncio

from kms2.config.source_processing import InstructionGovernanceSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.instruction import Instruction
from kms2.core.model.source_processing.pedagogical import StatementDraft
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_processing.instruction_governance import (
    InstructionGovernanceNode,
)


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls += 1
        return [1 for _ in texts]


class Judge:
    def __init__(self) -> None:
        self.calls = []

    async def aforward(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return kwargs['statement_blocks'][0].content == 'exercise one'


def _state() -> SourceProcessingState:
    contents = [
        'instruction one',
        'exercise one',
        'exercise detail',
        'definition',
        'instruction two',
        'later exercise',
    ]
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(contents)
    ]
    return SourceProcessingState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[SourcePage(index=0, blocks=blocks)],
        instructions=[
            Instruction(member_block_uuids=['block-0']),
            Instruction(member_block_uuids=['block-4']),
        ],
        statements=[
            StatementDraft(
                member_block_uuids=['block-1', 'block-2'], is_exercise=True
            ),
            StatementDraft(member_block_uuids=['block-3'], is_exercise=False),
            StatementDraft(member_block_uuids=['block-5'], is_exercise=True),
        ],
    )


def test_instruction_governance_projects_only_accepted_exercise_statements():
    judge = Judge()
    state = _state()
    original_pages = state.split_pages

    result = asyncio.run(
        InstructionGovernanceNode(
            judge,
            InstructionGovernanceSettings(),
            token_counters=(_Counter(),),
        ).run(state)
    )

    assert [
        item.governed_statement_uuids for item in result['instructions']
    ] == [[state.statements[0].uuid], []]
    assert len(judge.calls) == 2
    assert [
        context.content for context in judge.calls[0]['statement_blocks']
    ] == ['exercise one', 'exercise detail']
    assert [
        context.content for context in judge.calls[1]['statement_blocks']
    ] == ['later exercise']
    assert state.split_pages is original_pages
    assert state.statements[0].is_exercise is True


def test_governance_clears_old_ids_without_counting_when_no_eligible_statements():
    counter = _Counter()
    judge = Judge()
    state = _state()
    state.statements = []
    state.instructions = [
        Instruction(
            member_block_uuids=['block-0'],
            governed_statement_uuids=['stale-statement'],
        )
    ]

    result = asyncio.run(
        InstructionGovernanceNode(
            judge,
            InstructionGovernanceSettings(),
            token_counters=(counter,),
        ).run(state)
    )

    assert result['instructions'][0].governed_statement_uuids == []
    assert counter.calls == 0
    assert judge.calls == []
