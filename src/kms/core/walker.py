from kms.core import models


def estimate_tokens(node: models.ASTNode) -> int:
    return len(node.content or '') // 4 + 1


def window_from(nodes: list[models.ASTNode], cursor: int, budget: int) -> int:
    """Return the exclusive end index of a look-ahead window starting at `cursor`:
    whole nodes up to the soft token budget, always at least one node."""
    i, accumulated = cursor, 0
    node_count = len(nodes)
    while i < node_count:
        token_count = estimate_tokens(nodes[i])
        if i > cursor and accumulated + token_count > budget:
            break
        accumulated += token_count
        i += 1
    return i
