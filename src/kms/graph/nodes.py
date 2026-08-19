"""Deterministic uuid and property builders for node vertices."""

from typing import Any

from kms.core import identity, models

NODE_LABEL = 'Node'
SOURCE_LABEL = 'Source'


def node_uuid(source: str, index: int) -> str:
    """Returns the deterministic uuid for one AST node."""
    return identity.node_uuid(source, index)


def source_uuid(source: str) -> str:
    """Returns the deterministic uuid for a source document."""
    return identity.source_uuid(source)


def source_properties(
    source: str, metadata: dict[str, Any] | None = None
) -> dict:
    """Builds the property dict used to persist a source vertex.

    Args:
        source: The source key.
        metadata: Optional extra source properties.

    Returns:
        The source properties with None values dropped.
    """
    properties = {
        **(metadata or {}),
        'uuid': source_uuid(source),
        'key': source,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def node_label(node: models.Node) -> str | None:
    """Returns the Neo4j label for a node's type, title-cased."""
    ntype = node.type
    return ntype.title() if ntype else None


def node_properties(
    node: models.Node, source: str, embedding: list[float] | None = None
) -> dict:
    """Builds the property dict used to persist one AST node.

    Args:
        node: The AST node.
        source: The source key the node belongs to.
        embedding: Optional vector embedding for the node content.

    Returns:
        The node properties with None values dropped.
    """
    properties = {
        'uuid': node_uuid(source, node.id),
        'source': source_uuid(source),
        'type': node.type,
        'content': node.content,
        'index': node.id,
        'document_index': node.document_index,
        'image_path': node.image_path,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def block_key(block: list[int]) -> str:
    """Serializes a member id block into a stable string key."""
    return '#'.join(str(node_id) for node_id in block)
