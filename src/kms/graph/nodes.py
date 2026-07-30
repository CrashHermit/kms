"""
Graph representation of the structural node stream — the one part of the
model we're sure of.

The extractor emits domain-agnostic structural nodes (paragraph / math /
code / list / table / image / caption / header / bibliographic / note — the
``core.models.ASTNode`` subclasses),
the seam merger flattens them into the global ordered stream, and that
stream is the provenance layer every later graph tier points back to. This
module maps a ``core.models.ASTNode`` onto its Neo4j form. It invents no new
node types — the vocabulary is ``core.models.NodeType`` — and it stays free
of the neo4j driver (pure mapping); the driver lives in ``graph.db``.

Representation: every structural node carries the shared ``:Node`` label AND
its per-type label (``:Node:Math``, ``:Node:Paragraph``, … — Neo4j nodes
hold multiple labels). ``:Node`` spans the whole stream (the uuid key and
traversal attach here); the per-type label makes ``MATCH (n:Math)`` a native
label scan with no property index. The kind is also kept as a ``type``
property, mirroring ``models.ASTNode.type`` for readback.

Identity: a node's in-document int id collides across books, so the stable
vertex key is a uuid (the HANDOFF's deferred decision), and the int is
demoted to an ``index`` provenance property. The uuid is DETERMINISTIC —
uuid5 over ``(source, index)`` — so re-persisting the same book MERGEs onto
the same vertices instead of duplicating them, and ``source`` disambiguates
the same index across different books. Callers pass ``source`` (the book
identity); ``index`` is the node's id, which is always assigned by the time
the flat stream reaches the graph tier.

Source root: each book is a ``:Source`` vertex (deterministic
``source_uuid``) that roots its stream via ``(:Source)-[:HEAD]->(:Node)`` to
the first node — walk ``:NEXT`` from there to read the book in order. Every
``:Node`` also carries a ``source`` property (the source uuid) so
book-scoped lookups are one indexed hop, not a chain walk. Book metadata
(title, author, …) is open: a required ``key`` plus whatever the caller
supplies, not a fixed schema.
"""

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kms.core import models

NODE_LABEL = 'Node'
SOURCE_LABEL = 'Source'


def node_uuid(source: str, index: int) -> str:
    """Stable, deterministic vertex key for a structural node.

    A uuid5 over the book ``source`` and the node's document-order
    ``index``. Deterministic so a re-run MERGEs rather than duplicates;
    ``source`` keeps the same index in two different books distinct.

    Args:
        source: The stable book identity.
        index: The node's document-order id.

    Returns:
        The node's hex uuid.
    """
    return uuid5(NAMESPACE_URL, f'{source}#{index}').hex


def source_uuid(source: str) -> str:
    """Stable, deterministic vertex key for a book/source.

    A uuid5 over the source key. This is the ``:Source`` node's uuid and the
    value each ``:Node`` carries in its ``source`` property.

    Args:
        source: The stable book identity.

    Returns:
        The source's hex uuid.
    """
    return uuid5(NAMESPACE_URL, source).hex


def source_properties(
    source: str, metadata: dict[str, Any] | None = None
) -> dict:
    """The Neo4j property map for the ``:Source`` node.

    Holds its uuid, the ``key`` (the stable source string), and any
    caller-supplied book metadata (title/author/…). ``uuid``/``key`` are
    authoritative — metadata can't clobber them — and None values are
    dropped.

    Args:
        source: The stable book identity.
        metadata: Optional book metadata to merge in.

    Returns:
        The property map, with None values omitted.
    """
    props = {**(metadata or {}), 'uuid': source_uuid(source), 'key': source}
    return {key: value for key, value in props.items() if value is not None}


def node_label(node: models.ASTNode) -> str | None:
    """The per-type label for a structural node, from its class name.

    Args:
        node: The structural node.

    Returns:
        The label (e.g. ``Math``), or None if the class carries no suffix.
    """
    return type(node).__name__.removesuffix('Node')


def node_properties(node: models.ASTNode, source: str) -> dict:
    """The Neo4j property map for one structural node.

    Holds its stable uuid, the structural type, the markdown content, and
    provenance (document-order ``index`` + originating ``segment_index``).
    Precondition: ``node.id`` is set (true once the stream is flattened).

    Args:
        node: The structural node, with ``id`` assigned.
        source: The stable book identity.

    Returns:
        The property map, with None values omitted rather than written as
        nulls.
    """
    props = {
        'uuid': node_uuid(source, node.id),
        'source': source_uuid(source),  # links back to the :Source node
        'type': node.kind,
        'content': node.content,
        'index': node.id,
        'segment_index': node.segment_index,
    }
    return {key: value for key, value in props.items() if value is not None}
