import asyncio

from kms.config.source_pedagogical import InstructionGovernanceSettings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.context import SourceBlockContext
from kms.core.model.page import SourcePage
from kms.core.model.source_pedagogical.instruction import (
    Instruction,
    InstructionGovernanceDecision,
    InstructionGovernanceInput,
)
from kms.core.model.source_pedagogical.pedagogical import StatementDraft
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.node.source_pedagogical.instruction_governance import (
    InstructionGovernanceNode,
)


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls += 1
        return [1 for _ in texts]


class _Judge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def acall(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        items: list[InstructionGovernanceInput],
        content_images: list,
    ) -> list[InstructionGovernanceDecision]:
        self.calls.append(
            {'instruction_blocks': instruction_blocks, 'items': items}
        )
        return [
            InstructionGovernanceDecision(
                position=item.position,
                governs=item.statement_blocks[0].material.text.endswith(
                    'accepted'
                ),
            )
            for item in reversed(items)
        ]


def _state() -> SourcePedagogicalState:
    contents = [f'context {index}' for index in range(16)]
    contents[0] = 'directive one'
    contents[10] = 'directive two'
    for index in (2, 4, 6, 12, 14):
        contents[index] = (
            f'exercise {index} accepted'
            if index in (2, 6, 12)
            else f'exercise {index} rejected'
        )
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(contents)
    ]
    return SourcePedagogicalState(
        source_uuid='source-1',
        split_pages=[SourcePage(index=0, blocks=blocks)],
        instructions=[
            Instruction(
                member_block_uuids=['block-0'],
                governed_statement_uuids=['stale-one'],
            ),
            Instruction(
                member_block_uuids=['block-10'],
                governed_statement_uuids=['stale-two'],
            ),
        ],
        statements=[
            StatementDraft(
                member_block_uuids=[f'block-{index}'], is_exercise=True
            )
            for index in (2, 4, 6, 12, 14)
        ]
        + [StatementDraft(member_block_uuids=['block-8'], is_exercise=False)],
    )


def _run(
    state: SourcePedagogicalState, batch_size: int
) -> tuple[dict, _Judge, _Counter]:
    judge = _Judge()
    counter = _Counter()
    settings = InstructionGovernanceSettings(batch_size=batch_size)
    result = asyncio.run(
        InstructionGovernanceNode(
            judge,
            settings,
            budget=TokenBudget(MaterialTokenEstimator(counter), 10_000),
            token_counters=(MaterialTokenEstimator(counter),),
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ).run(state)
    )
    return result, judge, counter


def test_governance_batches_independent_candidates_and_restores_statement_ids():
    state = _state()
    original_blocks = [
        block.model_copy(deep=True) for block in state.split_pages[0].blocks
    ]
    batched, judge, _ = _run(state, 16)

    assert [
        item.governed_statement_uuids for item in batched['instructions']
    ] == [
        [state.statements[0].uuid, state.statements[2].uuid],
        [state.statements[3].uuid],
    ]
    assert len(judge.calls) == 2
    assert [item.position for item in judge.calls[0]['items']] == [0, 1, 2]
    assert [item.position for item in judge.calls[1]['items']] == [0, 1]
    assert [
        item.statement_blocks[0].material.text
        for item in judge.calls[0]['items']
    ] == [
        'exercise 2 accepted',
        'exercise 4 rejected',
        'exercise 6 accepted',
    ]
    assert len(judge.calls[0]['instruction_blocks']) == 1
    assert all(
        item.context_before or item.context_after
        for item in judge.calls[0]['items']
    )
    assert (
        judge.calls[0]['items'][0].context_before
        != judge.calls[0]['items'][1].context_before
    )
    assert state.split_pages[0].blocks == original_blocks

    singleton, singleton_judge, _ = _run(state, 1)
    assert len(singleton_judge.calls) == 5
    assert [
        item.governed_statement_uuids for item in singleton['instructions']
    ] == [item.governed_statement_uuids for item in batched['instructions']]


def test_governance_clears_old_ids_without_counting_or_inference_when_empty():
    state = _state()
    state.statements = []
    state.instructions = [
        Instruction(
            member_block_uuids=['block-0'],
            governed_statement_uuids=['stale-statement'],
        )
    ]
    result, judge, counter = _run(state, 16)
    assert result['instructions'][0].governed_statement_uuids == []
    assert counter.calls == 0
    assert judge.calls == []
