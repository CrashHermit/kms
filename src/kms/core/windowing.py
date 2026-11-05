"""Select ordered source context and pack whole items."""

from collections.abc import Sequence

from kms.core.model.block import SourceBlock
from kms.core.model.context import SourceBlockWindow
from kms.core.tokenization.budget import InputBudgetExceeded


def select_window(
    blocks: list[SourceBlock],
    target_positions: list[int],
    *,
    token_counts: Sequence[int],
    backward_budget: int | None = None,
    forward_budget: int | None = None,
    target_budget: int = 0,
    max_targets: int | None = None,
) -> SourceBlockWindow:
    """Select canonical blocks and budgeted context around target positions."""
    target = [blocks[position] for position in target_positions]

    target_start = target_positions[0]
    target_end = target_positions[-1]
    if target_budget > 0:
        target_cost = sum(
            token_counts[position] for position in target_positions
        )
        for position in range(target_end + 1, len(blocks)):
            if max_targets is not None and len(target) >= max_targets:
                break
            cost = token_counts[position]
            if target_cost + cost > target_budget:
                break
            target.append(blocks[position])
            target_cost += cost
            target_end = position

    context_before: list[SourceBlock] = []
    if backward_budget is not None and backward_budget > 0:
        before_cost = 0
        for position in range(target_start - 1, -1, -1):
            block = blocks[position]
            cost = token_counts[position]
            if before_cost + cost > backward_budget:
                break
            context_before.append(block)
            before_cost += cost
        context_before.reverse()

    context_after: list[SourceBlock] = []
    if forward_budget is not None and forward_budget > 0:
        after_cost = 0
        for position in range(target_end + 1, len(blocks)):
            block = blocks[position]
            cost = token_counts[position]
            if after_cost + cost > forward_budget:
                break
            context_after.append(block)
            after_cost += cost

    return SourceBlockWindow(
        context_before=context_before,
        target=target,
        context_after=context_after,
    )


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
