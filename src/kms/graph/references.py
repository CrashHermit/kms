"""
Graph representation of cross-entity references — the reference edges + their canonical-entity targets.

The referencers fill each entity's ``refs`` (a list of ``core.models.Reference`` — a named target,
what kind of thing it is, and the open relation that holds). This module maps those onto Neo4j: each
reference becomes an edge ``(:Entity)-[:REFERENCES {relation}]->(:Entity:Canonical)`` onto a
**canonical** node, and this module is the pure planning half (uuids, normalization, canonical/edge
rows) — the driver lives in ``graph.db`` and the writes in ``graph.writer``.

The edge property is an **open, LLM-named** relation, not one of nine math tactics
(``docs/GENERALIZATION.md``, step 2). It stays a property on a single ``:REFERENCES`` type rather than
becoming the relationship type itself: an open vocabulary as the type would make every traversal
guess the type set, whereas one type + a property keeps "what does this cite?" a single-hop query and
leaves the naming free.

The canonical is the design's connective node (``docs/UNIFIED-KG.md``): references from any entity —
and later, any book — that name the same target converge on ONE canonical, so the reference graph
doesn't fragment across sources. It is a full ``:Entity`` carrying the ``:Canonical`` role label, NOT
a disjoint kind — so a canonical and the book-specific mentions of it share the ``:Entity`` label and
the ``type`` property (open, as everywhere: definition / theorem / law / …), while
``MATCH (:Canonical)`` isolates the hubs. Two consequences shape the identity:

* The canonical uuid is **global, NOT source-scoped** (unlike node/entity uuids). That is the whole
  point: a reference to "Set" in book A and one in book B must land on the same canonical.
* The uuid is a deterministic uuid5 over ``(kind, normalized target)``, where normalization lowercases
  and collapses whitespace. That gives cheap exact-name clustering for free; the real semantic dedup
  (embed → judge, and tying a canonical to the corpus's own mention entities via ``:REALIZES``) is a
  later tier that refines these canonicals, it does not replace them.

References uniformly target canonicals even when the target also exists as a mention ``:Entity`` in the
same book — mixing "edge to the canonical" and "edge straight to the local mention" would reintroduce
the fragmentation the canonical exists to prevent. Linking a canonical to the matching in-corpus
mention (``:REALIZES``) is the later dedup tier's job, not the referencer's.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph.entities import entity_uuid


def normalize_target(kind: str, target: str) -> str:
    """The clustering key for a reference target: its kind plus the lowercased, whitespace-collapsed
    name. Two references that name the same thing with trivial spacing/case differences share a key
    (and therefore a canonical); genuine paraphrases stay distinct until the semantic dedup tier merges
    them."""
    return f'{kind.strip().lower()}#{" ".join(target.split()).lower()}'


def canonical_uuid(kind: str, target: str) -> str:
    """Stable, deterministic vertex key for a canonical entity: uuid5 over ``(kind, normalized
    target)``. Global on purpose — NO ``source`` prefix — so the same target from different books/
    entities resolves to the same canonical. The ``canonical#`` segment keeps it disjoint from the
    source-scoped node and mention-entity uuids (which key on ``source#…``)."""
    return uuid5(
        NAMESPACE_URL, f'canonical#{normalize_target(kind, target)}'
    ).hex


def canonical_properties(kind: str, target: str) -> dict:
    """The Neo4j property map for one canonical: its global uuid, the ``type`` (the entity type, from
    the reference kind — a property, like every mention's type), and the ``name`` as written (the
    first spelling that minted it — cosmetic; the uuid is what identity keys on). No ``source``: a
    canonical is corpus-level, not book-scoped."""
    return {
        'uuid': canonical_uuid(kind, target),
        'type': kind.strip().lower(),
        'name': target.strip(),
    }


def canonical_rows(entities: list[models.Entity]) -> list[dict]:
    """The unique canonical property maps across every reference in the overlay, de-duplicated by
    uuid. One flat list: a canonical's type is a property, not a label, so one batched MERGE writes
    them all."""
    seen: dict[str, dict] = {}
    for entity in entities:
        for ref in entity.refs:
            props = canonical_properties(ref.kind, ref.target)
            seen.setdefault(props['uuid'], props)
    return list(seen.values())


def reference_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{entity, canonical, relation}`` rows for the ``:REFERENCES`` edges: one per (entity,
    reference). The citing entity's uuid is source-scoped (it is an in-corpus mention); the canonical
    uuid is global."""
    return [
        {
            'entity': entity_uuid(source, entity.id),
            'canonical': canonical_uuid(ref.kind, ref.target),
            'relation': ref.relation,
        }
        for entity in entities
        for ref in entity.refs
    ]
