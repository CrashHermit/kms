from kms2.core.context_window import (
    estimate_text_tokens,
    estimate_tokens,
    project_block,
    select_window,
)
from kms2.core.model import (
    BlockType,
    SourceBlock,
    SourceBlockContext,
    SourceContextWindow,
    VisualAsset,
)


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


def test_source_context_window_defaults_are_independent():
    first = SourceContextWindow()
    second = SourceContextWindow()

    first.context_before.append(
        SourceBlockContext(block_type=BlockType.PARAGRAPH)
    )

    assert first.context_before
    assert second.context_before == []
    assert second.target == []
    assert second.context_after == []


def test_project_block_preserves_model_facing_fields_only():
    block = _block(
        7,
        'source text',
        BlockType.IMAGE,
        assets=[VisualAsset(path='first.png'), VisualAsset(path='second.png')],
    )

    projected = project_block(block)

    assert projected.block_type is BlockType.IMAGE
    assert projected.content == 'source text'
    assert projected.asset_paths == ['first.png', 'second.png']
    assert set(projected.model_dump()) == {
        'block_type',
        'content',
        'asset_paths',
    }


def test_select_window_returns_ordered_target_and_independent_context_sides():
    blocks = [_block(index, 'abc') for index in range(8)]

    window = select_window(
        blocks,
        [2, 3, 4],
        backward_budget=2,
        forward_budget=2,
    )

    assert [item.content for item in window.context_before] == ['abc', 'abc']
    assert [item.content for item in window.target] == ['abc', 'abc', 'abc']
    assert [item.content for item in window.context_after] == ['abc', 'abc']


def test_select_window_expands_targets_with_optional_target_budget():
    blocks = [_block(index, 'abc') for index in range(8)]

    window = select_window(
        blocks,
        [2],
        backward_budget=2,
        target_budget=3,
        forward_budget=2,
    )

    assert [item.content for item in window.target] == ['abc', 'abc', 'abc']
    assert [item.content for item in window.context_before] == ['abc', 'abc']
    assert [item.content for item in window.context_after] == ['abc', 'abc']


def test_select_window_disables_optional_context_sides():
    blocks = [_block(index, 'abc') for index in range(3)]

    window = select_window(blocks, [1])

    assert window.context_before == []
    assert [item.content for item in window.target] == ['abc']
    assert window.context_after == []


def test_select_window_includes_exact_budget_boundaries_and_stops_on_oversized_nearby_block():
    blocks = [_block(index) for index in range(6)]

    exact = select_window(
        blocks,
        [3],
        backward_budget=2,
        forward_budget=2,
    )
    assert len(exact.context_before) == 2
    assert len(exact.context_after) == 2

    zero = select_window(
        blocks,
        [3],
        backward_budget=0,
        forward_budget=0,
    )
    assert zero.context_before == []
    assert zero.context_after == []

    oversized = select_window(
        [_block(0), _block(1, 'aaaaaaaa'), _block(2, 'target')],
        [2],
        backward_budget=2,
        forward_budget=0,
    )
    assert oversized.context_before == []


def test_token_estimates_ignore_asset_paths():
    visual_context = SourceBlockContext(
        block_type=BlockType.IMAGE,
        asset_paths=['a-very-long-image-path-that-is-not-tokenized.png'],
    )

    assert estimate_text_tokens(None) == 1
    assert estimate_text_tokens('abcd') == 2
    assert estimate_text_tokens('abcde') == 2
    assert estimate_tokens(visual_context) == 1
