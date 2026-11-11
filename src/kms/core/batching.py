"""Pack ordered whole items by precomputed token costs."""

from collections.abc import Sequence

from kms.core.tokenization.budget import InputBudgetExceeded


def pack_items[Item](
    items: Sequence[Item],
    *,
    token_counts: Sequence[int],
    token_budget: int,
    max_items: int | None = None,
) -> list[list[Item]]:
    """Pack ordered whole items by precomputed integer costs."""
    batches: list[list[Item]] = []
    batch: list[Item] = []
    batch_cost = 0
    for index, (item, item_cost) in enumerate(
        zip(items, token_counts, strict=True)
    ):
        exceeds_budget = batch and batch_cost + item_cost > token_budget
        exceeds_item_limit = (
            batch and max_items is not None and len(batch) >= max_items
        )
        if exceeds_budget or exceeds_item_limit:
            batches.append(batch)
            batch = []
            batch_cost = 0
        if not batch and item_cost > token_budget:
            raise InputBudgetExceeded(
                f'Item at index {index} cannot fit as a singleton'
            )
        batch.append(item)
        batch_cost += item_cost
    if batch:
        batches.append(batch)
    return batches
