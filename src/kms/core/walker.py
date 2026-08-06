"""Look-ahead windowing over the flat node stream.

The window stages (instruction finder, PCF, splitter) all walk the stream in
token-budgeted chunks rather than node counts, so a window holds roughly the
same amount of text whatever the node sizes are. The fixed-window passes
(atomic facts) cut the same stream into adjacent
fixed windows of whole nodes and share one view model for what a windowed
pass sees. All helpers are pure.
"""

from pydantic import BaseModel

from kms.core import models


def estimate_text_tokens(text: str | None) -> int:
    """A rough token estimate for a text string: length ÷ 4, clamped.

    Args:
        text: The text to size.

    Returns:
        The estimated token count, at least 1.
    """
    return len(text or '') // 4 + 1


def estimate_tokens(node: models.ASTNode) -> int:
    """A rough token estimate for a node: its content's token count.

    Args:
        node: The node to size.

    Returns:
        The estimated token count, at least 1.
    """
    return estimate_text_tokens(node.content)


def window_from(nodes: list[models.ASTNode], cursor: int, budget: int) -> int:
    """Return the exclusive end index of a look-ahead window.

    The window holds whole nodes up to the soft token budget, and always at
    least one node so the walk cannot stall.

    Args:
        nodes: The flat node stream.
        cursor: The window's inclusive start index.
        budget: The soft token budget for the window.

    Returns:
        The window's exclusive end index.
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


class WindowNode(BaseModel):
    """One node of a fixed window as a windowed pass sees it.

    ``node_id`` is the stream's node id — fixed-window passes attribute
    their output back to the node it came from, so attribution travels as
    the id rather than a window-local position. (The look-ahead passes
    define their own position-based view: they select WITHIN a window,
    they do not attribute output to a node.)
    """

    node_id: int
    type: str
    content: str | None = None


def fixed_windows_with_context(
    nodes: list[models.ASTNode],
    budget: int,
    backward_budget: int,
    forward_budget: int,
) -> list[tuple[list[models.ASTNode], str | None, str | None]]:
    """Cut the node stream into adjacent fixed windows, each with the
    text immediately before and after it.

    ``image`` nodes are dropped, every other node with content is
    eligible, windows are adjacent and non-overlapping, and a window
    always contains at least one node. Around each central window the
    stream is walked backward from its first node (within
    ``backward_budget`` tokens) and forward from its last node (within
    ``forward_budget`` tokens) to collect surrounding context. The three
    budgets are independent so the parts of a window can be tuned
    separately. Context is placement-only — a windowed pass must not
    extract output from it.

    Args:
        nodes: The flat node stream.
        budget: The central window's soft token budget.
        backward_budget: Token budget for the context before the window.
        forward_budget: Token budget for the context after the window.

    Returns:
        One ``(window, before, after)`` per window, in document order.
        ``before``/``after`` are the joined context text, or None when
        nothing qualifies.
    """
    eligible = [
        (index, node)
        for index, node in enumerate(nodes)
        if node.id is not None
        and node.content
        and node.content.strip()
        and node.type != 'image'
    ]

    windows: list[tuple[list[models.ASTNode], str | None, str | None]] = []
    current: list[tuple[int, models.ASTNode]] = []
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
    nodes: list[models.ASTNode],
    entries: list[tuple[int, models.ASTNode]],
    backward_budget: int,
    forward_budget: int,
) -> tuple[list[models.ASTNode], str | None, str | None]:
    """Build one window's output: its nodes plus before/after context.

    Args:
        nodes: The flat node stream (for the context walks).
        entries: ``(original_index, node)`` pairs in the window.
        backward_budget: Token budget for the context before the window.
        forward_budget: Token budget for the context after the window.

    Returns:
        The ``(window, before, after)`` triple.
    """
    window = [node for _, node in entries]
    first_index = entries[0][0]
    last_index = entries[-1][0]
    before = content_before(nodes, first_index, backward_budget)
    after = content_after(nodes, last_index, forward_budget)
    return window, before, after


def content_before(
    nodes: list[models.ASTNode], cursor: int, budget: int
) -> str | None:
    """Return the concatenated content of nodes before *cursor*,
    walking backward within *budget* tokens.

    Nodes with no content are skipped. The result is in document order.
    Returns None when there is nothing to show.

    Args:
        nodes: The flat node stream.
        cursor: The exclusive end index (the position of the focus node).
        budget: The soft token budget for the backward window.

    Returns:
        The joined content of the qualifying preceding nodes, or None.
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
    nodes: list[models.ASTNode], cursor: int, budget: int
) -> str | None:
    """Return the concatenated content of nodes after *cursor*,
    walking forward within *budget* tokens.

    Nodes with no content are skipped. Returns None when there is nothing
    to show.

    Args:
        nodes: The flat node stream.
        cursor: The exclusive start index (the position of the focus node).
        budget: The soft token budget for the forward window.

    Returns:
        The joined content of the qualifying following nodes, or None.
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
"""A position-indexed content string for context walking."""


def context_around(
    items: list[ContextItem],
    cursor: int,
    backward_budget: int = 500,
    forward_budget: int = 200,
) -> tuple[str | None, str | None]:
    """Return the before/after context around *cursor* in a
    position-indexed content list.

    Walks backward from the cursor (``index < cursor``) within
    *backward_budget* tokens, then forward (``index > cursor``) within
    *forward_budget*. Items at or past each budget boundary are excluded.
    Both results are joined with blank-line separators.

    Args:
        items: ``(position, content)`` entries in document order.
        cursor: The position of the focus item — excluded from both windows.
        backward_budget: Soft token budget for the preceding window.
        forward_budget: Soft token budget for the following window.

    Returns:
        The ``(before, after)`` pair, each possibly None.
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
