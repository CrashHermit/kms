"""
Graph representation of the structural node stream — the one part of the model we're sure of.

The extractor emits domain-agnostic structural nodes (paragraph / math / code / list / table /
image / caption / header — ``core.NodeType``), the seam merger flattens them into the global
ordered stream, and that stream is the provenance layer every later graph tier points back to.
This module maps a ``core.ASTNode`` onto its Neo4j form. It invents no new node types — the
vocabulary is ``core.NodeType`` — and it stays free of the neo4j driver (pure mapping); the
driver lives in ``graph.db``.

Representation: every structural node carries the shared ``:Node`` label AND its per-type label
(``:Node:Math``, ``:Node:Paragraph``, … — Neo4j nodes hold multiple labels). ``:Node`` spans the
whole stream (the uuid key and traversal attach here); the per-type label makes ``MATCH (n:Math)``
a native label scan with no property index. The kind is also kept as a ``type`` property, mirroring
``ASTNode.type`` for readback.

Identity: a node's in-document int id collides across books, so the stable vertex key is a uuid
(the HANDOFF's deferred decision), and the int is demoted to an ``index`` provenance property.
The uuid is DETERMINISTIC — uuid5 over ``(source, index)`` — so re-persisting the same book MERGEs
onto the same vertices instead of duplicating them, and ``source`` disambiguates the same index
across different books. Callers pass ``source`` (the book identity); ``index`` is the node's id,
which is always assigned by the time the flat stream reaches the graph tier.

Source root: each book is a ``:Source`` vertex (deterministic ``source_uuid``) that roots its
stream via ``(:Source)-[:HEAD]->(:Node)`` to the first node — walk ``:NEXT`` from there to read the
book in order. Every ``:Node`` also carries a ``source`` property (the source uuid) so book-scoped
lookups are one indexed hop, not a chain walk. Book metadata (title, author, …) is open: a required
``key`` plus whatever the caller supplies, not a fixed schema.
"""

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kms.core.models import ASTNode

NODE_LABEL = "Node"
SOURCE_LABEL = "Source"


def node_uuid(source: str, index: int) -> str:
    """Stable, deterministic vertex key for a structural node: uuid5 over the book ``source``
    and the node's document-order ``index``. Deterministic so a re-run MERGEs rather than
    duplicates; ``source`` keeps the same index in two different books distinct."""
    return uuid5(NAMESPACE_URL, f"{source}#{index}").hex


def source_uuid(source: str) -> str:
    """Stable, deterministic vertex key for a book/source: uuid5 over the source key. This is the
    ``:Source`` node's uuid and the value each ``:Node`` carries in its ``source`` property."""
    return uuid5(NAMESPACE_URL, source).hex


def source_properties(source: str, metadata: dict[str, Any] | None = None) -> dict:
    """The Neo4j property map for the ``:Source`` node: its uuid, the ``key`` (the stable source
    string), and any caller-supplied book metadata (title/author/…). ``uuid``/``key`` are
    authoritative — metadata can't clobber them — and None values are dropped."""
    props = {**(metadata or {}), "uuid": source_uuid(source), "key": source}
    return {key: value for key, value in props.items() if value is not None}


def node_label(node: ASTNode) -> str | None:
    """The per-type label for a structural node (``NodeType.MATH`` -> ``"Math"``), or None if the
    node has no type. Applied ALONGSIDE the base ``:Node`` label, never instead of it. The
    NodeType values are single lowercase words, so capitalizing yields a valid Neo4j label."""
    return node.type.value.capitalize() if node.type else None


def node_properties(node: ASTNode, source: str) -> dict:
    """The Neo4j property map for one structural node: its stable uuid, the structural type,
    the markdown content, and provenance (document-order ``index`` + originating ``seg_index``).
    None-valued fields (e.g. an unset ``role``) are omitted rather than written as nulls.
    Precondition: ``node.id`` is set (true once the stream is flattened)."""
    props = {
        "uuid": node_uuid(source, node.id),
        "source": source_uuid(source),  # links back to the :Source node
        "type": node.type.value if node.type else None,
        "content": node.content,
        "index": node.id,
        "seg_index": node.seg_index,
        "role": node.role,
    }
    return {key: value for key, value in props.items() if value is not None}
