"""
Graph representation of the structural node stream — the one part of the model we're sure of.

The extractor emits domain-agnostic structural nodes (paragraph / math / code / list / table /
image / caption / header — ``core.NodeType``), the seam merger flattens them into the global
ordered stream, and that stream is the provenance layer every later graph tier points back to.
This module maps a ``core.ASTNode`` onto its Neo4j form. It invents no new node types — the
vocabulary is ``core.NodeType`` — and it stays free of the neo4j driver (pure mapping); the
driver lives in ``graph.db``.

Representation: one ``:Node`` label carrying the structural kind as a ``type`` property (mirrors
``ASTNode.type`` one-to-one). Per-type labels (``:Math``, ``:Paragraph``, …) are a possible later
refinement if querying wants them; kept to a single label for now.

Identity: a node's in-document int id collides across books, so the stable vertex key is a uuid
(the HANDOFF's deferred decision), and the int is demoted to an ``index`` provenance property.
The uuid is DETERMINISTIC — uuid5 over ``(source, index)`` — so re-persisting the same book MERGEs
onto the same vertices instead of duplicating them, and ``source`` disambiguates the same index
across different books. Callers pass ``source`` (the book identity); ``index`` is the node's id,
which is always assigned by the time the flat stream reaches the graph tier.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core.models import ASTNode

NODE_LABEL = "Node"


def node_uuid(source: str, index: int) -> str:
    """Stable, deterministic vertex key for a structural node: uuid5 over the book ``source``
    and the node's document-order ``index``. Deterministic so a re-run MERGEs rather than
    duplicates; ``source`` keeps the same index in two different books distinct."""
    return uuid5(NAMESPACE_URL, f"{source}#{index}").hex


def node_properties(node: ASTNode, source: str) -> dict:
    """The Neo4j property map for one structural node: its stable uuid, the structural type,
    the markdown content, and provenance (document-order ``index`` + originating ``seg_index``).
    None-valued fields (e.g. an unset ``role``) are omitted, matching how they're dropped from
    ``nodes.json``. Precondition: ``node.id`` is set (true once the stream is flattened)."""
    props = {
        "uuid": node_uuid(source, node.id),
        "type": node.type.value if node.type else None,
        "content": node.content,
        "index": node.id,
        "seg_index": node.seg_index,
        "role": node.role,
    }
    return {key: value for key, value in props.items() if value is not None}
