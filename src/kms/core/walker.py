from pydantic import BaseModel

from kms.core import models


def estimate_text_tokens(text: str | None) -> int:
    return len(text or '') // 4 + 1


def estimate_tokens(node: models.ASTNode) -> int:
    return estimate_text_tokens(node.content)


def window_from(nodes: list[models.ASTNode], cursor: int, budget: int) -> int:
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
    node_id: int
    type: str
    content: str | None = None


def fixed_windows_with_context(
    nodes: list[models.ASTNode],
    budget: int,
    backward_budget: int,
    forward_budget: int,
) -> list[tuple[list[models.ASTNode], str | None, str | None]]:
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
    window = [node for _, node in entries]
    first_index = entries[0][0]
    last_index = entries[-1][0]
    before = content_before(nodes, first_index, backward_budget)
    after = content_after(nodes, last_index, forward_budget)
    return window, before, after


def content_before(
    nodes: list[models.ASTNode], cursor: int, budget: int
) -> str | None:
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
