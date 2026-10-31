import asyncio
import json

from kms2.config.inference import ContextWindowSettings
from kms2.config.source_pedagogical import ExerciseFinderSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.page import SourcePage
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseStartDecision,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.node.source_pedagogical.exercise_finder import ExerciseFinderNode


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
        self.calls: list[list[ExerciseClassificationInput]] = []

    async def acall(self, *, items: list[ExerciseClassificationInput]):
        self.calls.append(items)
        return [
            ExerciseStartDecision(
                position=item.position,
                is_exercise_start=item.target_block.content in self.starts,
            )
            for item in reversed(items)
        ]


class _BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries
        self.calls: list[
            tuple[SourceBlockContext, list[ExerciseClassificationInput]]
        ] = []

    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[ExerciseClassificationInput],
    ):
        self.calls.append((start_block, items))
        return [
            ExerciseBoundaryDecision(
                position=item.position,
                is_exercise_boundary=item.target_block.content
                in self.boundaries,
            )
            for item in reversed(items)
        ]


def _state(
    *pages: tuple[tuple[BlockType, str], ...],
    instructions: list[Instruction] | None = None,
) -> SourcePedagogicalState:
    source_pages = [
        SourcePage(
            index=page_index,
            blocks=[
                SourceBlock(
                    uuid=f'block-{page_index}-{block_index}',
                    block_type=block_type,
                    content=content,
                )
                for block_index, (block_type, content) in enumerate(blocks)
            ],
        )
        for page_index, blocks in enumerate(pages)
    ]
    return SourcePedagogicalState(
        source_uuid='source-1',
        split_pages=source_pages,
        instructions=instructions or [],
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
) -> ExerciseFinderNode:
    start_counter = start_counter or _Counter()
    boundary_counter = boundary_counter or _Counter()
    settings = ExerciseFinderSettings(
        start_batch_size=batch_size,
        boundary_batch_size=batch_size,
        start_context_window=start_window
        or ContextWindowSettings(backward_budget=0, forward_budget=0),
        boundary_context_window=boundary_window
        or ContextWindowSettings(backward_budget=0, forward_budget=0),
    )
    return ExerciseFinderNode(
        _StartRouter(starts),
        _BoundaryRouter(boundaries),
        settings,
        start_budget=TokenBudget(start_counter, start_budget),
        boundary_budget=TokenBudget(boundary_counter, boundary_budget),
        start_token_counters=(start_counter,),
        boundary_token_counters=(boundary_counter,),
    )


def _members(result: dict) -> list[list[str]]:
    return [
        component.member_block_uuids
        for component in result['exercise_components']
    ]


def _blocks(
    *contents: tuple[BlockType, str],
) -> tuple[tuple[BlockType, str], ...]:
    return contents


def _text(content: str) -> tuple[BlockType, str]:
    return BlockType.PARAGRAPH, content


def test_batch_and_singleton_traversal_have_identical_membership():
    state = _state(
        _blocks(
            *map(_text, ('intro', 'Exercise 1', '1a', '1b', 'Exercise 2', '2a'))
        ),
        _blocks(*map(_text, ('2b', 'tail'))),
    )
    starts = {'Exercise 1', 'Exercise 2'}
    boundaries = {'Exercise 2'}
    batched = _node(starts, boundaries, batch_size=8)
    singleton = _node(starts, boundaries, batch_size=1)

    source_blocks = [
        block for page in state.split_pages for block in page.blocks
    ]
    original = [(block.uuid, block.content) for block in source_blocks]
    batch_result = asyncio.run(batched.run(state))
    singleton_result = asyncio.run(singleton.run(state))

    assert (
        _members(batch_result)
        == _members(singleton_result)
        == [
            ['block-0-1', 'block-0-2', 'block-0-3'],
            ['block-0-4', 'block-0-5', 'block-1-0', 'block-1-1'],
        ]
    )
    assert len(batched._start_router.calls) < len(singleton._start_router.calls)
    assert len(batched._boundary_router.calls) < len(
        singleton._boundary_router.calls
    )
    assert [(block.uuid, block.content) for block in source_blocks] == original


def test_instruction_owned_blocks_are_excluded_even_when_interleaved():
    state = _state(
        _blocks(
            *map(
                _text,
                (
                    'exercise start',
                    'owned one',
                    'continuation',
                    'owned two',
                    'tail',
                ),
            )
        ),
        instructions=[
            Instruction(member_block_uuids=['block-0-1', 'block-0-3'])
        ],
    )
    node = _node({'exercise start'}, set())

    result = asyncio.run(node.run(state))

    assert _members(result) == [['block-0-0', 'block-0-2', 'block-0-4']]
    assert all(
        item.target_block.content not in {'owned one', 'owned two'}
        for batch in node._start_router.calls
        for item in batch
    )


def test_equation_image_table_and_subparts_remain_in_exercise():
    state = _state(
        _blocks(
            _text('Exercise 4'),
            (BlockType.EQUATION, 'x^2 + y^2'),
            (BlockType.IMAGE, 'diagram'),
            (BlockType.TABLE, 'values'),
            _text('(a) solve'),
            _text('(b) explain'),
        )
    )
    result = asyncio.run(_node({'Exercise 4'}, set()).run(state))

    assert _members(result) == [[f'block-0-{index}' for index in range(6)]]


def test_earliest_out_of_order_boundary_is_exclusive_and_reconsidered():
    state = _state(
        _blocks(
            *map(
                _text,
                ('Exercise 1', 'early new exercise', 'later boundary', 'tail'),
            )
        )
    )
    node = _node(
        {'Exercise 1', 'early new exercise'},
        {'early new exercise', 'later boundary'},
    )

    result = asyncio.run(node.run(state))

    assert _members(result) == [['block-0-0'], ['block-0-1']]


def test_shared_topic_does_not_suppress_numbered_new_exercise_start():
    state = _state(
        _blocks(
            *map(
                _text,
                (
                    'Practice problems: fractions',
                    '1. Add these',
                    '2. Subtract these',
                ),
            )
        )
    )
    node = _node({'1. Add these', '2. Subtract these'}, {'2. Subtract these'})

    result = asyncio.run(node.run(state))

    assert _members(result) == [['block-0-1'], ['block-0-2']]


def test_all_false_batches_advance_and_empty_or_all_claimed_sources_skip_calls():
    no_hits = _node(set(), set(), batch_size=2)
    result = asyncio.run(
        no_hits.run(_state(_blocks(*map(_text, ('a', 'b', 'c', 'd', 'e')))))
    )
    assert result['exercise_components'] == []
    assert [len(items) for items in no_hits._start_router.calls] == [2, 2, 1]
    assert no_hits._boundary_router.calls == []

    empty = _node(set(), set())
    asyncio.run(empty.run(_state()))
    assert empty._start_router.calls == []
    assert empty._boundary_router.calls == []

    claimed = _node(set(), set())
    asyncio.run(
        claimed.run(
            _state(
                _blocks(_text('already owned')),
                instructions=[Instruction(member_block_uuids=['block-0-0'])],
            )
        )
    )
    assert claimed._start_router.calls == []
    assert claimed._boundary_router.calls == []


def test_distinct_context_counters_and_batch_items_have_independent_context():
    contents = ('before', 'Exercise 1', 'candidate A', 'candidate B')
    start_counter = _Counter(dict.fromkeys(contents, 1))
    boundary_counter = _Counter(dict.fromkeys(contents, 2))
    node = _node(
        {'Exercise 1'},
        set(),
        batch_size=3,
        start_counter=start_counter,
        boundary_counter=boundary_counter,
        start_window=ContextWindowSettings(backward_budget=1, forward_budget=1),
        boundary_window=ContextWindowSettings(
            backward_budget=2, forward_budget=2
        ),
    )
    state = _state(
        _blocks(
            *map(_text, ('before', 'Exercise 1', 'candidate A', 'candidate B'))
        )
    )

    asyncio.run(node.run(state))

    start_items = [item for batch in node._start_router.calls for item in batch]
    assert start_items[1].target_block.content == 'Exercise 1'
    assert [block.content for block in start_items[1].context_before] == [
        'before'
    ]
    assert start_items[2].target_block.content == 'candidate A'
    assert [block.content for block in start_items[2].context_before] == [
        'Exercise 1'
    ]
    assert start_counter.texts and boundary_counter.texts
    boundary_items = node._boundary_router.calls[0][1]
    assert boundary_items[0].target_block.content == 'candidate A'
    assert boundary_items[1].target_block.content == 'candidate B'
    assert [block.content for block in boundary_items[1].context_before] == [
        'candidate A'
    ]


def test_serialized_budget_splits_batches_and_counts_shared_boundary_anchor_once():
    start_counter = _Counter()
    boundary_counter = _Counter()
    state = _state(
        _blocks(
            *map(
                _text,
                (
                    'Exercise start with context',
                    'candidate one',
                    'candidate two',
                ),
            )
        )
    )
    anchor = SourceBlockContext(
        block_type=BlockType.PARAGRAPH,
        content='Exercise start with context',
    )
    items = [
        ExerciseClassificationInput(
            position=index,
            context_before=[
                anchor
                if index == 0
                else SourceBlockContext(
                    block_type=BlockType.PARAGRAPH, content='candidate one'
                )
            ],
            target_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH, content=content
            ),
            context_after=[],
        )
        for index, content in enumerate(('candidate one', 'candidate two'))
    ]
    costs = [
        len(
            json.dumps(
                [item.model_dump(mode='json')],
                ensure_ascii=False,
                separators=(',', ':'),
            )
        )
        for item in items
    ]
    anchor_cost = len(
        json.dumps(
            {'start_block': anchor.model_dump(mode='json')},
            ensure_ascii=False,
            separators=(',', ':'),
        )
    )
    node = _node(
        {'Exercise start with context'},
        set(),
        batch_size=8,
        start_counter=start_counter,
        boundary_counter=boundary_counter,
        start_budget=1000,
        boundary_budget=sum(costs) + anchor_cost - 1,
        boundary_window=ContextWindowSettings(
            backward_budget=40, forward_budget=0
        ),
    )

    asyncio.run(node.run(state))

    assert [len(batch) for batch in node._start_router.calls] == [3]
    assert [len(batch) for _, batch in node._boundary_router.calls] == [1, 1]
    anchor_json = json.dumps(
        {'start_block': anchor.model_dump(mode='json')},
        ensure_ascii=False,
        separators=(',', ':'),
    )
    assert anchor_json in boundary_counter.texts


def test_boundary_without_hit_closes_component_at_end_of_source():
    node = _node({'Exercise 1'}, set(), batch_size=2)
    result = asyncio.run(
        node.run(
            _state(
                _blocks(*map(_text, ('Exercise 1', 'one'))),
                _blocks(*map(_text, ('two', 'three'))),
            )
        )
    )
    assert _members(result) == [
        ['block-0-0', 'block-0-1', 'block-1-0', 'block-1-1']
    ]
