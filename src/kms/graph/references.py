"""
Graph representation of cross-entity references — the reference edges + their canonical-entity targets.

The per-type referencers fill each entity's ``refs`` (a list of ``core.Reference`` — a named target,
its kind, and the tactic role it plays). This module maps those onto Neo4j: each reference becomes an
edge ``(:Entity)-[:REFERENCES {tactic}]->(:Entity:Canonical)`` onto a **canonical** node, and this
module is the pure planning half (uuids, normalization, canonical/edge rows) — the driver lives in
``graph.db`` and the writes in ``graph.writer``.

The canonical is the design's connective node (``docs/UNIFIED-KG.md``): references from any entity —
and later, any book — that name the same target converge on ONE canonical, so the reference graph
doesn't fragment across sources. It is a full ``:Entity`` carrying the ``:Canonical`` role label (and
its per-type label, from the reference kind), NOT a disjoint kind — so ``MATCH (:Definition)`` sees a
canonical alongside the book-specific mentions of it, while ``MATCH (:Canonical)`` isolates the hubs.
Two consequences shape the identity:

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

from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from kms.core.models import Entity
from kms.graph.entities import entity_uuid


def normalize_target(kind: str, target: str) -> str:
    """The clustering key for a reference target: its kind plus the lowercased, whitespace-collapsed
    name. Two references that name the same thing with trivial spacing/case differences share a key
    (and therefore a canonical); genuine paraphrases stay distinct until the semantic dedup tier merges
    them."""
    return f"{kind.strip().lower()}#{' '.join(target.split()).lower()}"


def canonical_uuid(kind: str, target: str) -> str:
    """Stable, deterministic vertex key for a canonical entity: uuid5 over ``(kind, normalized
    target)``. Global on purpose — NO ``source`` prefix — so the same target from different books/
    entities resolves to the same canonical. The ``canonical#`` segment keeps it disjoint from the
    source-scoped node and mention-entity uuids (which key on ``source#…``)."""
    return uuid5(NAMESPACE_URL, f"canonical#{normalize_target(kind, target)}").hex


def canonical_type_label(kind: str) -> str:
    """The per-type label for a canonical, from the reference kind (``"definition"`` -> ``"Definition"``),
    applied alongside the base ``:Entity`` and role ``:Canonical`` labels — so a canonical is typed like
    the mentions it stands in for. The reference kinds are single lowercase words, so capitalizing yields
    a valid Neo4j label."""
    return kind.strip().lower().capitalize()


def canonical_properties(kind: str, target: str) -> dict:
    """The Neo4j property map for one canonical: its global uuid, the ``type`` (the entity type, from the
    reference kind), and the ``name`` as written (the first spelling that minted it — cosmetic; the uuid
    is what identity keys on). No ``source``: a canonical is corpus-level, not book-scoped."""
    return {
        "uuid": canonical_uuid(kind, target),
        "type": kind.strip().lower(),
        "name": target.strip(),
    }


def canonical_batches(entities: list[Entity]) -> dict[str, list[dict]]:
    """The unique canonical property maps across every reference in the overlay — de-duplicated by uuid
    and grouped by per-type label, so each label is one batched MERGE (mirrors ``entity_batches``)."""
    seen: dict[str, tuple[str, dict]] = {}
    for entity in entities:
        for ref in entity.refs:
            props = canonical_properties(ref.kind, ref.target)
            seen[props["uuid"]] = (canonical_type_label(ref.kind), props)
    batches: dict[str, list[dict]] = defaultdict(list)
    for label, props in seen.values():
        batches[label].append(props)
    return dict(batches)


def reference_rows(entities: list[Entity], source: str) -> list[dict]:
    """The ``{entity, canonical, tactic}`` rows for the ``:REFERENCES`` edges: one per (entity,
    reference). The citing entity's uuid is source-scoped (it is an in-corpus mention); the canonical
    uuid is global."""
    return [
        {
            "entity": entity_uuid(source, entity.id),
            "canonical": canonical_uuid(ref.kind, ref.target),
            "tactic": ref.tactic,
        }
        for entity in entities
        for ref in entity.refs
    ]
