import pytest

from kms.core.batching import token_batches


def test_token_batches_preserve_order_and_budget() -> None:
    items = ['a', 'b', 'c']
    batches = token_batches(
        items,
        token_cost={'a': 1000, 'b': 2000, 'c': 2000}.__getitem__,
        token_budget=4096,
    )
    assert batches == [['a', 'b'], ['c']]


def test_token_batches_emit_oversized_item() -> None:
    assert token_batches(
        ['small', 'large'],
        token_cost={'small': 100, 'large': 5000}.__getitem__,
        token_budget=4096,
    ) == [['small'], ['large']]


@pytest.mark.parametrize(
    ('cost', 'budget'),
    [(0, 4096), (-1, 4096), (1, 0)],
)
def test_token_batches_reject_invalid_budget_or_cost(
    cost: int, budget: int
) -> None:
    with pytest.raises(ValueError):
        token_batches(['item'], token_cost=lambda _: cost, token_budget=budget)


def test_token_batches_return_empty_for_empty_input() -> None:
    assert token_batches([], token_cost=lambda _: 1, token_budget=1) == []
