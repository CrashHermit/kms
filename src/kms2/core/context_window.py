"""Select ordered, budgeted context around source blocks."""

from kms2.core.model.context import SourceBlockContext, SourceContextWindow
from kms2.core.model.source import SourceBlock


def project_block(source_block: SourceBlock) -> SourceBlockContext:
    """Project a canonical source block to model-facing context data."""

    return SourceBlockContext(
        block_type=source_block.block_type,
        content=source_block.content,
        asset_paths=[asset.path for asset in source_block.assets],
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
    backward_budget: int,
    target_budget: int,
    forward_budget: int,
) -> SourceContextWindow:
    target = [project_block(blocks[position]) for position in target_positions]

    target_start = target_positions[0]
    target_end = target_positions[-1]

    context_before: list[SourceBlockContext] = []
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


def select_cursor_window(
    blocks: list[SourceBlock],
    cursor: int,
    *,
    backward_budget: int,
    target_budget: int,
    forward_budget: int,
) -> tuple[int, SourceContextWindow]:
    """Select a budgeted target span beginning at the supplied cursor."""

    target_positions = [cursor]
    target_cost = estimate_tokens(project_block(blocks[cursor]))
    for position in range(cursor + 1, len(blocks)):
        context = project_block(blocks[position])
        cost = estimate_tokens(context)
        if target_cost + cost > target_budget:
            break
        target_positions.append(position)
        target_cost += cost

    window = select_window(
        blocks,
        target_positions,
        backward_budget=backward_budget,
        target_budget=target_budget,
        forward_budget=forward_budget,
    )
    return target_positions[-1] + 1, window
