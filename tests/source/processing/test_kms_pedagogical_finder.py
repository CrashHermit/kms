import asyncio

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.context import SourceBlockContext
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.material import ExerciseComponent, Instruction
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.processing.pedagogical_finder.models import (
    PedagogicalBoundaryDecision,
    PedagogicalClassificationInput,
    PedagogicalStartDecision,
)
from kms.source.processing.pedagogical_finder.node import PedagogicalFinderNode
from kms.source.processing.pedagogical_finder.settings import (
    PedagogicalFinderSettings,
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
        self.calls: list[list[PedagogicalClassificationInput]] = []

    async def acall(self, *, items: list[PedagogicalClassificationInput]):
        self.calls.append(items)
        return [
            PedagogicalStartDecision(
                position=item.position,
                is_pedagogical_start=item.target_block.material.text
                in self.starts,
            )
            for item in reversed(items)
        ]


class _BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries
        self.calls: list[
            tuple[SourceBlockContext, list[PedagogicalClassificationInput]]
        ] = []

    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[PedagogicalClassificationInput],
    ):
        self.calls.append((start_block, items))
        return [
            PedagogicalBoundaryDecision(
                position=item.position,
                is_pedagogical_boundary=item.target_block.material.text
                in self.boundaries,
            )
            for item in reversed(items)
        ]


def _state(
    *pages: tuple[tuple[BlockType, str], ...],
    instructions: list[Instruction] | None = None,
    exercises: list[ExerciseComponent] | None = None,
) -> SourceProcessingState:
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
    return SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source-1', key='source.pdf'),
        split_pages=source_pages,
        instructions=instructions or [],
        exercise_components=exercises or [],
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
) -> PedagogicalFinderNode:
    start_counter = start_counter or _Counter()
    boundary_counter = boundary_counter or _Counter()
    settings = PedagogicalFinderSettings(
        start_batch_size=batch_size,
        boundary_batch_size=batch_size,
        start_input_token_budget=start_budget,
        boundary_input_token_budget=boundary_budget,
        safety_margin_tokens=0,
        start_context_window=start_window
        or ContextWindowSettings(backward_budget=0, forward_budget=0),
        boundary_context_window=boundary_window
        or ContextWindowSettings(backward_budget=0, forward_budget=0),
    )
    return PedagogicalFinderNode(
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


def _members(result: dict) -> list[list[str]]:
    return [
        component.member_block_uuids
        for component in result['pedagogical_components']
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
            *map(_text, ('intro', 'definition', 'definition detail', 'tail'))
        ),
        _blocks(
            (BlockType.EQUATION, 'x^2 + y^2'),
            _text('theorem'),
            _text('theorem detail'),
            _text('worked example'),
            _text('example step'),
            _text('procedure'),
            _text('step one'),
        ),
    )
    starts = {'definition', 'theorem', 'worked example', 'procedure'}
    boundaries = {'theorem', 'worked example', 'procedure'}
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
            ['block-0-1', 'block-0-2', 'block-0-3', 'block-1-0'],
            ['block-1-1', 'block-1-2'],
            ['block-1-3', 'block-1-4'],
            ['block-1-5', 'block-1-6'],
        ]
    )
    assert len(batched._start_router.calls) < len(singleton._start_router.calls)
    assert len(batched._boundary_router.calls) < len(
        singleton._boundary_router.calls
    )
    assert [(block.uuid, block.content) for block in source_blocks] == original


def test_instruction_and_exercise_members_are_excluded_when_interleaved():
    state = _state(
        _blocks(
            *(
                _text(content)
                for content in (
                    'component start',
                    'instruction owned',
                    'continuation',
                    'exercise owned',
                    'tail',
                )
            )
        ),
        instructions=[Instruction(member_block_uuids=['block-0-1'])],
        exercises=[ExerciseComponent(member_block_uuids=['block-0-3'])],
    )
    node = _node({'component start'}, set())

    result = asyncio.run(node.run(state))

    assert _members(result) == [['block-0-0', 'block-0-2', 'block-0-4']]
    assert all(
        item.target_block.material.text
        not in {'instruction owned', 'exercise owned'}
        for batch in node._start_router.calls
        for item in batch
    )


def test_earliest_out_of_order_boundary_is_exclusive_and_reconsidered():
    state = _state(
        _blocks(
            *map(
                _text,
                ('definition', 'new theorem', 'later boundary', 'tail'),
            )
        )
    )
    node = _node(
        {'definition', 'new theorem'},
        {'new theorem', 'later boundary'},
    )

    result = asyncio.run(node.run(state))

    assert _members(result) == [['block-0-0'], ['block-0-1']]


def test_numbered_components_exclude_shared_prose_and_keep_attached_parts():
    state = _state(
        _blocks(
            _text('shared instructions'),
            _text('Definition'),
            _text('definition continuation'),
            _text('Example 1'),
            (BlockType.EQUATION, 'x^2 = 4'),
            _text('(a) solve'),
            _text('Example 2'),
            _text('second example continuation'),
            _text('non-pedagogical closing prose'),
        )
    )
    node = _node(
        {'Definition', 'Example 1', 'Example 2'},
        {'Example 1', 'Example 2', 'non-pedagogical closing prose'},
    )

    result = asyncio.run(node.run(state))

    assert _members(result) == [
        ['block-0-1', 'block-0-2'],
        ['block-0-3', 'block-0-4', 'block-0-5'],
        ['block-0-6', 'block-0-7'],
    ]


def test_no_boundary_closes_component_at_end_of_multipage_source():
    node = _node({'procedure'}, set(), batch_size=2)
    result = asyncio.run(
        node.run(
            _state(
                _blocks(_text('procedure'), _text('step one')),
                _blocks(
                    (BlockType.EQUATION, 'a=b'),
                    _text('step two'),
                    _text('conclusion'),
                ),
            )
        )
    )

    assert _members(result) == [
        ['block-0-0', 'block-0-1', 'block-1-0', 'block-1-1', 'block-1-2']
    ]


def test_all_false_batches_advance_and_empty_or_all_claimed_sources_skip_calls():
    no_hits = _node(set(), set(), batch_size=2)
    result = asyncio.run(
        no_hits.run(_state(_blocks(*map(_text, ('a', 'b', 'c', 'd', 'e')))))
    )
    assert result['pedagogical_components'] == []
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
                _blocks(_text('instruction'), _text('exercise')),
                instructions=[Instruction(member_block_uuids=['block-0-0'])],
                exercises=[ExerciseComponent(member_block_uuids=['block-0-1'])],
            )
        )
    )
    assert claimed._start_router.calls == []
    assert claimed._boundary_router.calls == []


def test_context_costs_are_profile_specific_and_items_have_independent_context():
    contents = ('before', 'definition', 'candidate A', 'candidate B')
    start_counter = _Counter(dict.fromkeys(contents, 1))
    boundary_counter = _Counter(dict.fromkeys(contents, 2))
    node = _node(
        {'definition'},
        set(),
        batch_size=3,
        start_counter=start_counter,
        boundary_counter=boundary_counter,
        start_window=ContextWindowSettings(backward_budget=1, forward_budget=1),
        boundary_window=ContextWindowSettings(
            backward_budget=2, forward_budget=2
        ),
    )
    state = _state(_blocks(*map(_text, contents)))

    asyncio.run(node.run(state))

    start_items = [item for batch in node._start_router.calls for item in batch]
    definition_item = next(
        item
        for item in start_items
        if item.target_block.material.text == 'definition'
    )
    assert [
        block.material.text for block in definition_item.context_before
    ] == ['before']
    candidate_item = next(
        item
        for item in start_items
        if item.target_block.material.text == 'candidate A'
    )
    assert [block.material.text for block in candidate_item.context_before] == [
        'definition'
    ]
    boundary_items = node._boundary_router.calls[0][1]
    assert [item.position for item in boundary_items] == [0, 1]
    assert boundary_items[1].target_block.material.text == 'candidate B'
    assert [
        block.material.text for block in boundary_items[1].context_before
    ] == ['candidate A']
    assert start_counter.texts and boundary_counter.texts
    assert all(item.position >= 0 for item in start_items)
