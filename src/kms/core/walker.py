"""Look-ahead windowing over the flat node stream.

The window stages (instruction finder, PCF, splitter) all walk the stream in
token-budgeted chunks rather than node counts, so a window holds roughly the
same amount of text whatever the node sizes are. Both helpers are pure.
"""

from kms.core import models


def estimate_tokens(node: models.ASTNode) -> int:
    """A rough token estimate for a node: character count ÷ 4, clamped.

    Args:
        node: The node to size.

    Returns:
        The estimated token count, at least 1.
    """
    return len(node.content or '') // 4 + 1


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
