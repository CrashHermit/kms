"""
Graph representation of cross-entity references — the reference edges + their general-entity hubs.

The per-type referencers fill each entity's ``refs`` (a list of ``core.Reference`` — a named target,
its kind, and the tactic role it plays). This module maps those onto Neo4j: each reference becomes an
edge ``(:Entity)-[:REFERENCES {tactic}]->(:GeneralEntity)`` onto a **hub** node, and this module is
the pure planning half (uuids, normalization, edge/hub rows) — the driver lives in ``graph.db`` and
the writes in ``graph.writer``.

The hub is the design's connective node (AutoMathKG's "general entity"): references from any entity —
and later, any book — that name the same target converge on ONE hub, so the reference graph doesn't
fragment across sources. Two consequences shape the identity:

* The hub uuid is **global, NOT source-scoped** (unlike node/entity uuids). That is the whole point:
  a reference to "Set" in book A and one in book B must land on the same hub.
* The uuid is a deterministic uuid5 over ``(kind, normalized target)``, where normalization lowercases
  and collapses whitespace. That gives cheap exact-name clustering for free; the real semantic dedup
  (embed → judge, and tying a hub to the corpus's own canonical/mention entities) is a later tier that
  refines these hubs, it does not replace them.

References uniformly target hubs even when the target also exists as an ``:Entity`` in the same book —
mixing "edge to the hub" and "edge straight to the local mention" would reintroduce the fragmentation
the hub exists to prevent. Connecting a hub to the matching in-corpus entity is the later dedup tier's
job, not the referencer's.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core.models import Entity
from kms.graph.entities import entity_uuid

GENERAL_ENTITY_LABEL = "GeneralEntity"


def normalize_target(kind: str, target: str) -> str:
    """The clustering key for a reference target: its kind plus the lowercased, whitespace-collapsed
    name. Two references that name the same thing with trivial spacing/case differences share a key
    (and therefore a hub); genuine paraphrases stay distinct until the semantic dedup tier merges
    them."""
    return f"{kind.strip().lower()}#{' '.join(target.split()).lower()}"


def hub_uuid(kind: str, target: str) -> str:
    """Stable, deterministic vertex key for a general-entity hub: uuid5 over ``(kind, normalized
    target)``. Global on purpose — NO ``source`` prefix — so the same target from different books/
    entities resolves to the same hub. The ``generalentity#`` segment keeps it disjoint from node and
    entity uuids."""
    return uuid5(NAMESPACE_URL, f"generalentity#{normalize_target(kind, target)}").hex


def hub_properties(kind: str, target: str) -> dict:
    """The Neo4j property map for one hub: its global uuid, the target ``kind``, and the ``name`` as
    written (the first spelling that minted it — cosmetic; the uuid is what identity keys on)."""
    return {"uuid": hub_uuid(kind, target), "kind": kind.strip().lower(), "name": target.strip()}


def hub_batch(entities: list[Entity]) -> list[dict]:
    """The unique hub property maps across every reference in the overlay, de-duplicated by uuid — one
    batched MERGE mints them all."""
    hubs: dict[str, dict] = {}
    for entity in entities:
        for ref in entity.refs:
            props = hub_properties(ref.kind, ref.target)
            hubs[props["uuid"]] = props
    return list(hubs.values())


def reference_rows(entities: list[Entity], source: str) -> list[dict]:
    """The ``{entity, hub, tactic}`` rows for the ``:REFERENCES`` edges: one per (entity, reference).
    The citing entity's uuid is source-scoped (it is an in-corpus vertex); the hub uuid is global."""
    return [
        {
            "entity": entity_uuid(source, entity.id),
            "hub": hub_uuid(ref.kind, ref.target),
            "tactic": ref.tactic,
        }
        for entity in entities
        for ref in entity.refs
    ]
