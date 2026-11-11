import asyncio

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.context import SourceBlockContext
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.processing.instruction_finder.models import (
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionStartDecision,
)
from kms.source.processing.instruction_finder.node import InstructionFinderNode
from kms.source.processing.instruction_finder.settings import (
    InstructionFinderSettings,
)
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def __init__(self, costs: dict[str, int] | None = None) -> None:
        self.costs = costs or {}
        self.texts: list[str] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.texts.extend(texts)
        return [self.costs.get(text, len(text)) for text in texts]


class _StartRouter:
    def __init__(self, starts: set[str]) -> None:
        self.starts = starts
        self.calls: list[list[InstructionClassificationInput]] = []

    async def acall(self, *, items: list[InstructionClassificationInput]):
        self.calls.append(items)
        return [
            InstructionStartDecision(
                position=item.position,
                is_instruction_start=(
                    item.target_block.material.text in self.starts
                ),
            )
            for item in reversed(items)
        ]


class _BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries
        self.calls: list[
            tuple[SourceBlockContext, list[InstructionClassificationInput]]
        ] = []

    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[InstructionClassificationInput],
    ):
        self.calls.append((start_block, items))
        return [
            InstructionBoundaryDecision(
                position=item.position,
                is_instruction_boundary=(
                    item.target_block.material.text in self.boundaries
                ),
            )
            for item in reversed(items)
        ]


def _state(*pages: tuple[str, ...]) -> SourceProcessingState:
    source_pages = [
        SourcePage(
            index=page_index,
            blocks=[
                SourceBlock(
                    uuid=f'block-{page_index}-{block_index}',
                    block_type=BlockType.PARAGRAPH,
                    content=content,
                )
                for block_index, content in enumerate(contents)
            ],
        )
        for page_index, contents in enumerate(pages)
    ]
    return SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source-1', key='source.pdf'),
        split_pages=source_pages,
    )


def _node(
    starts: set[str],
    boundaries: set[str],
    *,
    batch_size: int = 16,
    start_counter: _Counter | None = None,
    boundary_counter: _Counter | None = None,
    start_budget: int = 100_000,
    boundary_budget: int = 100_000,
    start_window: ContextWindowSettings | None = None,
    boundary_window: ContextWindowSettings | None = None,
):
    start_counter = start_counter or _Counter()
    boundary_counter = boundary_counter or _Counter()
    settings = InstructionFinderSettings(
        start_batch_size=batch_size,
        boundary_batch_size=batch_size,
        start_context_window=start_window
        or ContextWindowSettings(backward_budget=0, forward_budget=0),
        boundary_context_window=boundary_window
        or ContextWindowSettings(backward_budget=0, forward_budget=0),
    )
    return InstructionFinderNode(
        _StartRouter(starts),
        _BoundaryRouter(boundaries),
        settings,
        start_budget=TokenBudget(
            MaterialTokenEstimator(start_counter), start_budget
        ),
        boundary_budget=TokenBudget(
            MaterialTokenEstimator(boundary_counter), boundary_budget
        ),
        start_token_counters=(MaterialTokenEstimator(start_counter),),
        boundary_token_counters=(MaterialTokenEstimator(boundary_counter),),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )


def _spans(result: dict) -> list[list[str]]:
    return [item.member_block_uuids for item in result['instructions']]


def test_batch_and_singleton_traversal_have_identical_spans():
    state = _state(
        ('cover', 'first directions', 'first detail', 'first end'),
        ('second directions', 'second detail', 'tail'),
    )
    source_blocks = [
        block for page in state.split_pages for block in page.blocks
    ]
    original = [(block.uuid, block.content) for block in source_blocks]
    starts = {'first directions', 'second directions'}
    boundaries = {'first end'}

    batched = _node(starts, boundaries, batch_size=8)
    singleton = _node(starts, boundaries, batch_size=1)
    batch_result = asyncio.run(batched.run(state))
    singleton_result = asyncio.run(singleton.run(state))

    assert (
        _spans(batch_result)
        == _spans(singleton_result)
        == [
            ['block-0-1', 'block-0-2'],
            ['block-1-0', 'block-1-1', 'block-1-2'],
        ]
    )
    assert len(batched._start_router.calls) < len(singleton._start_router.calls)
    assert len(batched._boundary_router.calls) < len(
        singleton._boundary_router.calls
    )
    assert [(block.uuid, block.content) for block in source_blocks] == original


def test_earliest_boundary_is_exclusive_and_reconsidered_as_next_start():
    state = _state(('first directions', 'early end', 'later end', 'tail'))
    node = _node(
        {'first directions', 'early end'},
        {'early end', 'later end'},
        batch_size=8,
    )

    result = asyncio.run(node.run(state))

    assert _spans(result) == [
        ['block-0-0'],
        ['block-0-1'],
    ]


def test_all_false_start_batches_advance_without_creating_instructions():
    node = _node(set(), set(), batch_size=2)

    result = asyncio.run(node.run(_state(('a', 'b', 'c', 'd', 'e'))))

    assert result['instructions'] == []
    assert [len(items) for items in node._start_router.calls] == [2, 2, 1]
    assert node._boundary_router.calls == []


def test_boundary_batches_close_at_end_of_source_when_no_boundary_is_found():
    node = _node({'directions'}, set(), batch_size=2)

    result = asyncio.run(
        node.run(_state(('directions', 'one', 'two'), ('three', 'four')))
    )

    assert _spans(result) == [
        [
            'block-0-0',
            'block-0-1',
            'block-0-2',
            'block-1-0',
            'block-1-1',
        ]
    ]


def test_start_and_boundary_contexts_use_their_own_token_counters():
    start_counter = _Counter({'before': 1, 'directions': 1, 'candidate': 1})
    boundary_counter = _Counter(
        {'before': 50, 'directions': 50, 'candidate': 50}
    )
    node = _node(
        {'directions'},
        set(),
        batch_size=1,
        start_counter=start_counter,
        boundary_counter=boundary_counter,
        start_window=ContextWindowSettings(backward_budget=2, forward_budget=0),
        boundary_window=ContextWindowSettings(
            backward_budget=75, forward_budget=0
        ),
    )

    asyncio.run(node.run(_state(('before', 'directions', 'candidate'))))

    start_items = [
        item
        for batch in node._start_router.calls
        for item in batch
        if item.target_block.material.text == 'directions'
    ]
    boundary_items = node._boundary_router.calls[0][1]
    assert [
        context.material.text for context in start_items[0].context_before
    ] == ['before']
    assert [
        context.material.text for context in boundary_items[0].context_before
    ] == ['directions']
    assert boundary_items[0].target_block.material.text == 'candidate'
    assert start_counter.texts.count('before') >= 1
    assert boundary_counter.texts.count('before') >= 1
