"""Token-budgeted batching helpers for bounded asynchronous work."""

from collections.abc import Callable, Iterable


def token_batches[T](
    items: Iterable[T],
    *,
    token_cost: Callable[[T], int],
    token_budget: int,
) -> list[list[T]]:
    """Partitions items into ordered batches under a token budget.

    An item whose cost exceeds the budget is emitted as a singleton because it
    cannot be split without changing the operation. Empty input returns no
    batches. Costs must be positive integers and the budget must be positive.
    """
    if token_budget <= 0:
        raise ValueError('token budget must be positive')
    batches: list[list[T]] = []
    current: list[T] = []
    current_cost = 0
    for item in items:
        cost = token_cost(item)
        if type(cost) is not int or cost <= 0:
            raise ValueError(
                f'token cost must be a positive integer, got {cost!r}'
            )
        if current and current_cost + cost > token_budget:
            batches.append(current)
            current = []
            current_cost = 0
        current.append(item)
        current_cost += cost
        if current_cost >= token_budget:
            batches.append(current)
            current = []
            current_cost = 0
    if current:
        batches.append(current)
    return batches
