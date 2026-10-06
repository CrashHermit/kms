"""Count model tokens and select ordered, budgeted evidence."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kms2.core.model.block import SourceBlock
from kms2.core.model.context import (
    SourceBlockContext,
    SourceContextWindow,
)


def project_block(source_block: SourceBlock) -> SourceBlockContext:
    """Project a canonical source block to model-facing context data."""

    return SourceBlockContext(
        block_type=source_block.block_type,
        content=source_block.content,
    )


@runtime_checkable
class TextTokenCounter(Protocol):
    """Count ordered text payloads with the consuming model's tokenizer."""

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return one raw token count per text, preserving input order."""
        ...


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Configured token counter and inclusive workload limit."""

    counter: TextTokenCounter
    token_limit: int


def fits_token_budget(*, token_count: int, threshold: int) -> bool:
    """Return whether the measured count is within the inclusive threshold."""
    return token_count <= threshold


class InputBudgetExceeded(ValueError):
    """One atomic input cannot fit within the configured request budget."""


def count_text_tokens(
    texts: Sequence[str | None], counters: Sequence[TextTokenCounter]
) -> list[int]:
    """Count each payload, taking conservative costs across consuming models.

    Counter results align with inputs. At least one counter is supplied for
    nonempty text; absent content has no raw tokens or automatic BOS/EOS.
    """
    normalized = [text or '' for text in texts]
    if not any(normalized):
        return [0] * len(normalized)
    counts = counters[0].count_texts(normalized)
    for counter in counters[1:]:
        other_counts = counter.count_texts(normalized)
        counts = [
            max(first, second)
            for first, second in zip(counts, other_counts, strict=True)
        ]
    return counts


def select_window(
    blocks: list[SourceBlock],
    target_positions: list[int],
    *,
    token_counts: Sequence[int],
    backward_budget: int | None = None,
    forward_budget: int | None = None,
    target_budget: int = 0,
) -> SourceContextWindow:
    """Select one or more targets and budgeted context around them."""
    target = [project_block(blocks[position]) for position in target_positions]

    target_start = target_positions[0]
    target_end = target_positions[-1]
    if target_budget > 0:
        target_cost = sum(
            token_counts[position] for position in target_positions
        )
        for position in range(target_end + 1, len(blocks)):
            cost = token_counts[position]
            if target_cost + cost > target_budget:
                break
            target.append(project_block(blocks[position]))
            target_cost += cost
            target_end = position

    context_before: list[SourceBlockContext] = []
    if backward_budget is not None and backward_budget > 0:
        before_cost = 0
        for position in range(target_start - 1, -1, -1):
            context = project_block(blocks[position])
            cost = token_counts[position]
            if before_cost + cost > backward_budget:
                break
            context_before.append(context)
            before_cost += cost
        context_before.reverse()

    context_after: list[SourceBlockContext] = []
    if forward_budget is not None and forward_budget > 0:
        after_cost = 0
        for position in range(target_end + 1, len(blocks)):
            context = project_block(blocks[position])
            cost = token_counts[position]
            if after_cost + cost > forward_budget:
                break
            context_after.append(context)
            after_cost += cost

    return SourceContextWindow(
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
