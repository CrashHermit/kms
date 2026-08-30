"""Traversal and span finding over the flattened node stream."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from kms.core import context_window, models

logger = logging.getLogger(__name__)


class Span(BaseModel):
    """An inclusive range of local window positions."""

    start: int = Field(
        description='First node position in the window (inclusive)'
    )
    end: int = Field(description='Last node position in the window (inclusive)')


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


def window_from(
    nodes: list[models.SourceNode], cursor: int, budget: int
) -> int:
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
        token_count = context_window.estimate_tokens(nodes[end])
        if end > cursor and accumulated + token_count > budget:
            break
        accumulated += token_count
        end += 1
    return end


async def find_spans(
    nodes: list[models.SourceNode],
    module: Any,
    readiness_module: Any,
    budget: int,
    max_budget: int,
) -> list[list[int]]:
    """Finds spans across the node stream via readiness-gated look-ahead windows.

    The readiness module certifies whether the current window is sufficient
    to determine every unit boundary that starts within it. An unready window
    doubles its token budget and retries from the identical cursor without
    consulting the span finder; at document end the finder always runs once
    and its output is authoritative. A ready window runs the finder exactly
    once, banks its spans, and advances past the final emitted span.

    ``module`` and ``readiness_module`` each expose
    ``aforward(current_nodes=...)``; the readiness module yields ``is_complete``
    and the finder yields ``list[Span]`` of zero-based local positions.

    Args:
        nodes: The node stream.
        module: The span finder module.
        readiness_module: The window-readiness module.
        budget: Initial look-ahead token budget per window.
        max_budget: Cap on window growth before the look-ahead limit fails.

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

            is_complete = await readiness_module.aforward(
                current_nodes=context_window.project_nodes(window)
            )

            if not is_complete and not reached_doc_end:
                if size >= max_budget:
                    raise ValueError(
                        f'window readiness span reaches the window edge at '
                        f'cursor {cursor} after reaching the '
                        f'{max_budget}-token look-ahead limit'
                    )
                logger.debug(
                    'grow: unready window at cursor %d; budget %d -> %d',
                    cursor,
                    size,
                    size * 2,
                )
                size *= 2
                continue

            spans = await module.aforward(
                current_nodes=context_window.project_nodes(window)
            )
            clean = validate_spans(spans, len(window))

            if not clean:
                cursor = end
                break

            for span in clean:
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
            cursor = cursor + clean[-1].end + 1
            break

    return spans_out
