"""Forward token-window selection over the flattened node stream."""

from kms.core import context_window, models


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
