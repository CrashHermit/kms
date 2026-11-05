from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.text import count_text_tokens
from kms.core.windowing import select_window


def _block(
    index: int,
    content: str | None = None,
    block_type: BlockType = BlockType.PARAGRAPH,
    *,
    assets: list[VisualAsset] | None = None,
) -> SourceBlock:
    return SourceBlock(
        uuid=f'block-{index}',
        block_type=block_type,
        content=content,
        crop_path=f'crop-{index}.png',
        crop_bbox=(index, index + 1, index + 2, index + 3),
        assets=assets or [],
    )


def test_select_window_returns_ordered_target_and_independent_context_sides():
    blocks = [_block(index, 'abc') for index in range(8)]

    window = select_window(
        blocks,
        [2, 3, 4],
        token_counts=[1] * len(blocks),
        backward_budget=2,
        forward_budget=2,
    )

    assert [item.content for item in window.context_before] == ['abc', 'abc']
    assert [item.content for item in window.target] == ['abc', 'abc', 'abc']
    assert [item.content for item in window.context_after] == ['abc', 'abc']
    assert window.context_before == blocks[:2]
    assert window.target == blocks[2:5]
    assert window.context_after == blocks[5:7]


def test_select_window_retains_visual_assets_and_crop_metadata():
    asset = VisualAsset(path='figure.png')
    visual = _block(1, None, BlockType.IMAGE, assets=[asset])
    blocks = [_block(0, 'before'), visual, _block(2, 'after')]

    window = select_window(
        blocks,
        [1],
        token_counts=[1, 0, 1],
        backward_budget=1,
        forward_budget=1,
    )

    assert window.target == [visual]
    assert window.target[0].assets == [asset]
    assert window.target[0].crop_path == visual.crop_path
    assert window.context_before == [blocks[0]]
    assert window.context_after == [blocks[2]]


def test_select_window_expands_targets_with_optional_target_budget():
    blocks = [_block(index, 'abc') for index in range(8)]

    window = select_window(
        blocks,
        [2],
        token_counts=[1] * len(blocks),
        backward_budget=2,
        target_budget=3,
        forward_budget=2,
    )

    assert [item.content for item in window.target] == ['abc', 'abc', 'abc']
    assert [item.content for item in window.context_before] == ['abc', 'abc']
    assert [item.content for item in window.context_after] == ['abc', 'abc']


def test_select_window_caps_expansion_without_consuming_following_context():
    blocks = [_block(index, str(index)) for index in range(6)]

    window = select_window(
        blocks,
        [1],
        token_counts=[1] * len(blocks),
        target_budget=5,
        forward_budget=3,
        max_targets=2,
    )

    assert [item.content for item in window.target] == ['1', '2']
    assert [item.content for item in window.context_after] == ['3', '4', '5']


def test_select_window_disables_optional_context_sides():
    blocks = [_block(index, 'abc') for index in range(3)]

    window = select_window(blocks, [1], token_counts=[1] * len(blocks))

    assert window.context_before == []
    assert [item.content for item in window.target] == ['abc']
    assert window.context_after == []


def test_select_window_includes_exact_budget_boundaries_and_stops_on_oversized_nearby_block():
    blocks = [_block(index, f'text-{index}') for index in range(6)]

    exact = select_window(
        blocks,
        [3],
        token_counts=[1] * len(blocks),
        backward_budget=2,
        forward_budget=2,
    )
    assert len(exact.context_before) == 2
    assert len(exact.context_after) == 2

    zero = select_window(
        blocks,
        [3],
        token_counts=[1] * len(blocks),
        backward_budget=0,
        forward_budget=0,
    )
    assert zero.context_before == []
    assert zero.context_after == []

    oversized = select_window(
        [_block(0), _block(1, 'aaaaaaaa'), _block(2, 'target')],
        [2],
        token_counts=[0, 3, 2],
        backward_budget=2,
        forward_budget=0,
    )
    assert oversized.context_before == []


class _Counter:
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.calls: list[list[str]] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls.append(texts)
        return self.counts


def test_multiple_model_costs_preserve_empty_positions_in_selection():
    first = _Counter([0, 2, 0, 5])
    second = _Counter([0, 4, 0, 3])
    blocks = [
        _block(0),
        _block(1, 'alpha'),
        _block(2, ''),
        _block(3, 'beta'),
    ]
    counts = count_text_tokens(
        [block.content for block in blocks], (first, second)
    )
    assert counts == [0, 4, 0, 5]
    window = select_window(blocks, [3], token_counts=counts, backward_budget=4)
    assert [item.content for item in window.context_before] == [
        None,
        'alpha',
        '',
    ]
    smaller = select_window(blocks, [3], token_counts=counts, backward_budget=3)
    assert [item.content for item in smaller.context_before] == ['']


def test_absent_content_has_zero_raw_cost_without_model_work():
    counter = _Counter([])
    assert count_text_tokens([], (counter,)) == []
    assert count_text_tokens([None, ''], (counter,)) == [0, 0]
    assert counter.calls == []
    blocks = [_block(index) for index in range(4)]
    disabled = select_window(
        blocks,
        [1],
        token_counts=[0] * len(blocks),
        backward_budget=0,
        forward_budget=0,
        target_budget=0,
    )
    assert len(disabled.target) == 1
    assert disabled.context_before == disabled.context_after == []
    enabled = select_window(
        blocks,
        [1],
        token_counts=[0] * len(blocks),
        backward_budget=1,
        forward_budget=1,
    )
    assert [item.content for item in enabled.context_before] == [None]
    assert [item.content for item in enabled.context_after] == [None, None]


def test_measured_costs_preserve_explicit_targets_and_exact_side_boundaries():
    blocks = [_block(index, str(index)) for index in range(4)]
    window = select_window(
        blocks,
        [2],
        token_counts=[2, 3, 5, 7],
        backward_budget=4,
        forward_budget=7,
        target_budget=0,
    )
    assert [item.content for item in window.context_before] == ['1']
    assert [item.content for item in window.target] == ['2']
    assert [item.content for item in window.context_after] == ['3']
    oversized_target = select_window(
        blocks, [2], token_counts=[2, 3, 5, 7], target_budget=1
    )
    assert [item.content for item in oversized_target.target] == ['2']
