"""Token-budgeted windows and span finding over the flattened node stream."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from kms.core import models

logger = logging.getLogger(__name__)


def estimate_text_tokens(text: str | None) -> int:
    """Rough token estimate for a text string (~4 chars per token)."""
    return len(text or '') // 4 + 1


def estimate_tokens(node: models.Node) -> int:
    """Rough token estimate for one node's content."""
    return estimate_text_tokens(node.content)


class WindowNode(BaseModel):
    """A node's projection within one window.

    ``position`` is the node's 0-based index within the window.
    """

    position: int
    type: str | None = None
    content: str | None = None
    image_path: str | None = None
    marker: str | None = None


class Span(BaseModel):
    """An inclusive range of local window positions."""

    start: int = Field(
        description='First local position of the span (inclusive).'
    )
    end: int = Field(description='Last local position of the span (inclusive).')


class Window(BaseModel):
    """One window of node views plus its surrounding context."""

    items: list[WindowNode] = Field(default_factory=list)
    before: str | None = None
    after: str | None = None


def node_views(nodes: list[models.Node]) -> list[WindowNode]:
    """Projects a run of nodes into window node views."""
    return [
        WindowNode(
            position=position,
            type=node.type,
            content=node.content,
            image_path=node.image_path,
        )
        for position, node in enumerate(nodes)
    ]


def validate_spans(spans: list[Span], window_size: int) -> list[Span]:
    """Validates finder spans without changing the model response.

    Finder output is an ownership contract. Invalid positions, ordering, or
    overlap are model errors and must be reported rather than repaired.
    """
    previous_end = -1
    for index, span in enumerate(spans):
        if not 0 <= span.start <= span.end < window_size:
            raise ValueError(
                f'invalid span {index} ({span.start}, {span.end}) for '
                f'window of {window_size} node(s)'
            )
        if span.start <= previous_end:
            raise ValueError(
                f'overlapping or out-of-order span {index} '
                f'({span.start}, {span.end}) after end {previous_end}'
            )
        previous_end = span.end
    return spans


def window_from(nodes: list[models.Node], cursor: int, budget: int) -> int:
    """Extends a window from cursor until the token budget is filled.

    The node at the cursor is always included even if it alone exceeds
    the budget; later nodes are added only while they fit.

    Args:
        nodes: The node stream.
        cursor: Index of the first node in the window.
        budget: Maximum accumulated token count.

    Returns:
        The exclusive end index of the window.
    """
    end, accumulated = cursor, 0
    node_count = len(nodes)
    while end < node_count:
        token_count = estimate_tokens(nodes[end])
        if end > cursor and accumulated + token_count > budget:
            break
        accumulated += token_count
        end += 1
    return end


def marked_window(
    nodes: list[models.Node],
    target_positions: list[int],
    backward_budget: int = 0,
    forward_budget: int = 0,
    marker: str = 'target',
) -> list[WindowNode]:
    """Builds one static marked window around one or more target nodes.

    The target nodes are always included. Context is added independently from
    the target span in each direction, so zero disables that side.
    """
    if not target_positions:
        raise ValueError('marked window requires at least one target position')
    if backward_budget < 0 or forward_budget < 0:
        raise ValueError('context budgets must be non-negative')
    if any(not 0 <= position < len(nodes) for position in target_positions):
        raise IndexError(
            f'target positions {target_positions} are outside '
            f'{len(nodes)} nodes'
        )

    target_set = set(target_positions)
    window_start = min(target_set)
    window_end = max(target_set) + 1

    accumulated = 0
    while window_start > 0:
        size = estimate_tokens(nodes[window_start - 1])
        if accumulated + size > backward_budget:
            break
        window_start -= 1
        accumulated += size

    accumulated = 0
    while window_end < len(nodes):
        size = estimate_tokens(nodes[window_end])
        if accumulated + size > forward_budget:
            break
        window_end += 1
        accumulated += size

    return [
        WindowNode(
            position=local_position,
            type=node.type,
            content=node.content,
            image_path=node.image_path,
            marker=(
                marker if window_start + local_position in target_set else None
            ),
        )
        for local_position, node in enumerate(nodes[window_start:window_end])
    ]


def fixed_windows_with_context(
    nodes: list[models.Node],
    budget: int,
    backward_budget: int,
    forward_budget: int,
) -> list[Window]:
    """Splits the node stream into token-budgeted content windows.

    Image nodes and empty content nodes are skipped when choosing what
    counts toward the budget; each finished window carries the preceding
    and following context strings up to their own budgets.

    Args:
        nodes: The full node stream.
        budget: Token budget for a window's own content.
        backward_budget: Token budget for the context before a window.
        forward_budget: Token budget for the context after a window.

    Returns:
        ``Window`` objects covering every eligible node.
    """
    eligible = [
        (index, node)
        for index, node in enumerate(nodes)
        if node.content
        and node.content.strip()
        and node.type != 'image'
    ]

    windows: list[Window] = []
    current: list[tuple[int, models.Node]] = []
    current_size = 0
    for entry in eligible:
        index, node = entry
        size = estimate_tokens(node)
        if current and current_size + size > budget:
            windows.append(
                _finish_window(nodes, current, backward_budget, forward_budget)
            )
            current = []
            current_size = 0
        current.append(entry)
        current_size += size
    if current:
        windows.append(
            _finish_window(nodes, current, backward_budget, forward_budget)
        )
    return windows


def _finish_window(
    nodes: list[models.Node],
    entries: list[tuple[int, models.Node]],
    backward_budget: int,
    forward_budget: int,
) -> Window:
    """Assembles the finished window with its surrounding context."""
    window = [node for _, node in entries]
    first_index = entries[0][0]
    last_index = entries[-1][0]
    before = content_before(nodes, first_index, backward_budget)
    after = content_after(nodes, last_index, forward_budget)
    return Window(items=node_views(window), before=before, after=after)


def content_before(
    nodes: list[models.Node], cursor: int, budget: int
) -> str | None:
    """Collects content immediately before a cursor, within a budget.

    Args:
        nodes: The node stream.
        cursor: The index to look back from (exclusive).
        budget: Maximum accumulated token count.

    Returns:
        The joined context, or None when there is none in budget.
    """
    accumulated = 0
    parts: list[str] = []
    for i in range(cursor - 1, -1, -1):
        node = nodes[i]
        content = node.content
        if not content or not content.strip():
            continue
        token_count = estimate_tokens(node)
        if accumulated + token_count > budget:
            break
        parts.append(content)
        accumulated += token_count
    if not parts:
        return None
    parts.reverse()
    return '\n\n'.join(parts)


def content_after(
    nodes: list[models.Node], cursor: int, budget: int
) -> str | None:
    """Collects content immediately after a cursor, within a budget.

    Args:
        nodes: The node stream.
        cursor: The index to look forward from (exclusive).
        budget: Maximum accumulated token count.

    Returns:
        The joined context, or None when there is none in budget.
    """
    accumulated = 0
    parts: list[str] = []
    for i in range(cursor + 1, len(nodes)):
        node = nodes[i]
        content = node.content
        if not content or not content.strip():
            continue
        token_count = estimate_tokens(node)
        if accumulated + token_count > budget:
            break
        parts.append(content)
        accumulated += token_count
    return '\n\n'.join(parts) or None


ContextItem = tuple[int, str]


def context_around(
    items: list[ContextItem],
    cursor: int,
    backward_budget: int = 500,
    forward_budget: int = 200,
) -> tuple[str | None, str | None]:
    """Builds before/after context around a cursor from positioned items.

    Args:
        items: ``(position, content)`` pairs in ascending position order.
        cursor: The position to center on.
        backward_budget: Token budget for content before the cursor.
        forward_budget: Token budget for content after the cursor.

    Returns:
        ``(before, after)`` context strings, either possibly None.
    """
    accumulated = 0
    before_parts: list[str] = []
    for position, content in items:
        if position >= cursor:
            break
        token_count = len(content) // 4 + 1
        if accumulated + token_count > backward_budget:
            break
        before_parts.append(content)
        accumulated += token_count

    accumulated = 0
    after_parts: list[str] = []
    for position, content in items:
        if position <= cursor:
            continue
        token_count = len(content) // 4 + 1
        if accumulated + token_count > forward_budget:
            break
        after_parts.append(content)
        accumulated += token_count

    content_before = '\n\n'.join(before_parts) or None
    content_after = '\n\n'.join(after_parts) or None
    return content_before, content_after


async def find_spans(
    nodes: list[models.Node],
    module: Any,
    budget: int,
    max_budget: int,
) -> list[list[int]]:
    """Finds spans across the node stream via growing look-ahead windows.

    Walks the stream in growing look-ahead windows; a span that reaches
    the window edge grows the window before being banked. ``module`` is
    any finder exposing ``aforward(current_nodes=...) -> list[Span]``.

    Args:
        nodes: The node stream.
        module: The finder module.
        budget: Initial look-ahead token budget per window.
        max_budget: Cap on window growth before banking as-is.

    Returns:
        A list of member position lists, one per span.
    """
    spans_out: list[list[int]] = []
    cursor, node_count = 0, len(nodes)

    while cursor < node_count:
        size = budget
        while True:
            end = window_from(nodes, cursor, size)
            window = nodes[cursor:end]
            reached_doc_end = end == node_count

            spans = await module.aforward(current_nodes=node_views(window))
            clean = validate_spans(spans, len(window))

            if not clean:
                cursor = end
                break
            bounded = [span for span in clean if span.end < len(window) - 1]

            if reached_doc_end:
                to_bank, advance = clean, end
            elif size >= max_budget:
                raise ValueError(
                    f'finder span reaches the window edge at cursor {cursor} '
                    f'after reaching the {max_budget}-token look-ahead limit'
                )
            elif bounded:
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                logger.debug(
                    'grow: sole span reaches the window edge at cursor %d; '
                    'budget %d -> %d',
                    cursor,
                    size,
                    size * 2,
                )
                size *= 2
                continue

            for span in to_bank:
                member_positions = [
                    cursor + position
                    for position in range(span.start, span.end + 1)
                ]
                if any(position >= len(nodes) for position in member_positions):
                    raise ValueError(
                        f'finder span ({span.start}, {span.end}) references '
                        f'a position outside node stream at cursor {cursor}'
                    )
                spans_out.append(member_positions)
            cursor = advance
            break

    return spans_out