"""
Graph representation of entity identity — the ``:REALIZES`` edges that tie an in-corpus mention to the
canonical hub it realizes (see ``docs/UNIFIED-KG.md``, "Roles: mention vs canonical").

The reference layer (``graph.references``) mints a global ``:Entity:Canonical`` per cited target, keyed
on ``(kind, normalized name)``, so citations across entities and books converge on ONE hub. But those
hubs start as **islands**: nothing connects a canonical back to the mention that actually defines or
states the concept — so a citation resolves to a bare name, not to the definition/theorem in the
corpus. This module draws that wire: a Definition/Theorem mention whose ``(type, title)`` normalizes to
an existing canonical's key gets a ``(:Entity:Mention)-[:REALIZES]->(:Entity:Canonical)`` edge. That is
what makes convergence *cross-corpus and traversable* — every book's mention of "vector space" realizes
the one canonical its citations also target, so "Theorem X references vector space" now resolves through
the shared hub to wherever vector space is actually defined.

Matching is the same cheap **nominal** clustering the canonical hubs already use — no embeddings. A
mention's ``title`` (its concept noun phrase, e.g. "Symmetric Group") is normalized and keyed with the
reference layer's own ``canonical_uuid``, so a mention and a reference that name the same thing land
on the same canonical uuid. Two design consequences, both intentional and matching the paper's model:

* **Every titled mention is a candidate.** There is no closed list of realizing types to filter on —
  the entity ``type`` is open, so a physics ``law`` realizes its canonical exactly as a math
  ``definition`` does. Filtering is left to the match: a type nothing cites (a ``problem``, typically,
  since tasks are not cited) mints a key no canonical carries, so it draws no edge.
* **The edge is drawn by MATCH, not minted.** ``realizes_rows`` emits a candidate per titled mention;
  the writer's ``MATCH`` on the canonical uuid does the filtering — a mention whose title was never
  cited finds no canonical and gets no edge (harmless), and a canonical with no realizing mention
  stays dangling (the "missing knowledge" case both papers cite). This keeps canonical minting the
  reference layer's sole job; ``:REALIZES`` only connects what already exists.

The semantic dedup tier (embedding fusion) later refines which mentions realize which canonical — this
is the deterministic floor it builds on, not a mechanism it replaces. Pure planning only (uuids,
matching, edge rows); the driver lives in ``graph.db`` and the writes in ``graph.writer``.
"""

from kms.core import models
from kms.graph.entities import entity_uuid
from kms.graph.references import canonical_uuid


def realizes_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{mention, canonical}`` uuid pairs for the ``:REALIZES`` edges — one candidate per typed,
    titled mention, keyed to the canonical its ``(type, title)`` names. De-duplicated by (mention,
    canonical). The mention uuid is source-scoped (an in-corpus entity); the canonical uuid is global,
    so a mention in any book resolves to the same hub its citations target. Whether the canonical
    actually exists is decided at write time by the ``MATCH`` (see the module docstring), so a row here
    is a candidate, not a guarantee of an edge."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        if not entity.type or not entity.title:
            continue
        mention = entity_uuid(source, entity.id)
        canonical = canonical_uuid(entity.type, entity.title)
        key = (mention, canonical)
        if key in seen:
            continue
        seen.add(key)
        rows.append({'mention': mention, 'canonical': canonical})
    return rows
