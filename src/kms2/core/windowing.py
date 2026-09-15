"""Select ordered, budgeted context around source blocks."""

from kms2.core.model import SourceBlock
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


def estimate_text_tokens(text: str | None) -> int:
    """Estimate text tokens using the repository's four-characters-per-token rule."""

    return len(text or '') // 4 + 1


def estimate_tokens(context: SourceBlockContext) -> int:
    """Estimate the text-context cost of one projected source block."""

    return estimate_text_tokens(context.content)


def select_window(
    blocks: list[SourceBlock],
    target_positions: list[int],
    *,
    backward_budget: int | None = None,
    forward_budget: int | None = None,
    target_budget: int = 0,
) -> SourceContextWindow:
    """Select one or more targets and budgeted context around them."""
    target = [project_block(blocks[position]) for position in target_positions]

    target_start = target_positions[0]
    target_end = target_positions[-1]
    target_cost = sum(estimate_tokens(context) for context in target)
    for position in range(target_end + 1, len(blocks)):
        context = project_block(blocks[position])
        cost = estimate_tokens(context)
        if target_cost + cost > target_budget:
            break
        target.append(context)
        target_cost += cost
        target_end = position

    context_before: list[SourceBlockContext] = []
    if backward_budget is not None:
        before_cost = 0
        for position in range(target_start - 1, -1, -1):
            context = project_block(blocks[position])
            cost = estimate_tokens(context)
            if before_cost + cost > backward_budget:
                break
            context_before.append(context)
            before_cost += cost
        context_before.reverse()

    context_after: list[SourceBlockContext] = []
    if forward_budget is not None:
        after_cost = 0
        for position in range(target_end + 1, len(blocks)):
            context = project_block(blocks[position])
            cost = estimate_tokens(context)
            if after_cost + cost > forward_budget:
                break
            context_after.append(context)
            after_cost += cost

    return SourceContextWindow(
        context_before=context_before,
        target=target,
        context_after=context_after,
    )


def window_from(blocks: list[SourceBlock], cursor: int, budget: int) -> int:
    """Return the exclusive endpoint of the next forward token window."""
    end = cursor
    accumulated = 0
    while end < len(blocks):
        token_count = estimate_text_tokens(blocks[end].content)
        if end > cursor and accumulated + token_count > budget:
            break
        accumulated += token_count
        end += 1
    return end
