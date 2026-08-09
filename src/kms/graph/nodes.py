from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kms.core import models

NODE_LABEL = 'Node'
SOURCE_LABEL = 'Source'


def node_uuid(source: str, index: int) -> str:
    return uuid5(NAMESPACE_URL, f'{source}#{index}').hex


def source_uuid(source: str) -> str:
    return uuid5(NAMESPACE_URL, source).hex


def source_properties(
    source: str, metadata: dict[str, Any] | None = None
) -> dict:
    properties = {
        **(metadata or {}),
        'uuid': source_uuid(source),
        'key': source,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def node_label(node: models.ASTNode) -> str | None:
    ntype = node.type
    return ntype.title() if ntype else None


def node_properties(
    node: models.ASTNode, source: str, embedding: list[float] | None = None
) -> dict:
    properties = {
        'uuid': node_uuid(source, node.id),
        'source': source_uuid(source),
        'type': node.type,
        'content': node.content,
        'index': node.id,
        'segment_index': node.segment_index,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def block_key(block: list[int]) -> str:
    return '#'.join(str(node_id) for node_id in block)

