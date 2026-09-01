"""Modality-neutral selection and projection of local node context."""

from pydantic import BaseModel, Field

from kms.core import models


class ContextNode(BaseModel):
    """A source node projected into one local context window."""

    position: int
    type: str | None = None
    content: str | None = None
    assets: list[models.VisualAsset] = Field(default_factory=list)
    marker: str | None = None


def estimate_text_tokens(text: str | None) -> int:
    """Rough token estimate for a text string (~4 chars per token)."""
    return len(text or '') // 4 + 1


def estimate_tokens(node: models.SourceNode) -> int:
    """Rough token estimate for one node's content."""
    return estimate_text_tokens(node.content)


def node_input(node: ContextNode, local_index: int = 0) -> models.NodeInput:
    """Projects one context node into a one-based model-facing record.

    Exposes only the text-only fields a DSPy signature may reference; it
    never surfaces ``ContextNode`` internals such as ``assets``,
    ``marker``, or source-global positions.
    """
    return models.NodeInput(
        index=local_index + 1,
        node_type=node.type or '',
        text=node.content or '',
    )


def project_nodes(nodes: list[models.SourceNode]) -> list[ContextNode]:
    """Projects source nodes into ordered local context nodes."""
    return [
        ContextNode(
            position=position,
            type=node.type,
            content=node.content,
            assets=node.assets.copy(),
        )
        for position, node in enumerate(nodes)
    ]


def select_around(
    nodes: list[models.SourceNode],
    target_positions: list[int],
    backward_budget: int = 0,
    forward_budget: int = 0,
    marker: str = 'target',
) -> list[ContextNode]:
    """Selects a marked local window around one or more target nodes.

    The target nodes are always included. Context is added independently from
    the target span in each direction, so zero disables that side.
    """
    if not target_positions:
        raise ValueError('context window requires at least one target')
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
        ContextNode(
            position=local_position,
            type=node.type,
            content=node.content,
            assets=node.assets.copy(),
            marker=(
                marker if window_start + local_position in target_set else None
            ),
        )
        for local_position, node in enumerate(nodes[window_start:window_end])
    ]


def nodes_before(
    nodes: list[models.SourceNode], cursor: int, budget: int
) -> list[models.SourceNode]:
    """Collects source nodes immediately before an exclusive cursor."""
    accumulated = 0
    selected: list[models.SourceNode] = []
    for index in range(cursor - 1, -1, -1):
        node = nodes[index]
        token_count = estimate_tokens(node)
        if accumulated + token_count > budget:
            break
        selected.append(node)
        accumulated += token_count
    selected.reverse()
    return selected


def nodes_after(
    nodes: list[models.SourceNode], cursor: int, budget: int
) -> list[models.SourceNode]:
    """Collects source nodes immediately after an exclusive cursor."""
    accumulated = 0
    selected: list[models.SourceNode] = []
    for index in range(cursor + 1, len(nodes)):
        node = nodes[index]
        token_count = estimate_tokens(node)
        if accumulated + token_count > budget:
            break
        selected.append(node)
        accumulated += token_count
    return selected


def select_target_context(
    nodes: list[models.SourceNode],
    position: int,
    before_budget: int,
    after_budget: int,
) -> tuple[
    list[ContextNode],
    ContextNode,
    list[ContextNode],
]:
    """Selects directional context around one target node.

    The target is always projected independently of either directional
    budget. Context nodes are collected in document order and projected with
    local positions for model-facing callers.
    """
    before = project_nodes(nodes_before(nodes, position, before_budget))
    target = project_nodes([nodes[position]])[0]
    after = project_nodes(nodes_after(nodes, position, after_budget))
    return before, target, after
