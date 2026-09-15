from kms2.core.model import BlockType, SourceBlock
from kms2.core.windowing import window_from


def _block(index: int, content: str) -> SourceBlock:
    return SourceBlock(
        uuid=f'block-{index}',
        block_type=BlockType.PARAGRAPH,
        content=content,
    )


def test_window_from_stops_before_first_later_block_over_budget():
    blocks = [_block(0, 'a' * 3), _block(1, 'b' * 4), _block(2, 'c' * 7)]

    assert window_from(blocks, cursor=0, budget=3) == 2


def test_window_from_budget_is_relative_to_middle_cursor():
    blocks = [
        _block(0, 'a' * 3),
        _block(1, 'b' * 3),
        _block(2, 'c' * 7),
        _block(3, 'd' * 3),
    ]

    assert window_from(blocks, cursor=1, budget=3) == 3


def test_window_from_includes_initial_block_when_it_exceeds_budget():
    blocks = [_block(0, 'a' * 20), _block(1, 'b' * 3)]

    assert window_from(blocks, cursor=0, budget=3) == 1


def test_window_from_includes_exact_budget_and_reaches_stream_end():
    blocks = [_block(0, 'a' * 3), _block(1, 'b' * 7), _block(2, 'c' * 11)]

    assert window_from(blocks, cursor=0, budget=6) == len(blocks)
