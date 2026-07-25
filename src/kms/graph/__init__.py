"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

Every module here is a pure ``models``→Neo4j mapping (uuids, property maps, edge rows) except two:
``db`` owns the driver and ``writer`` owns the I/O. That split is what lets the whole schema be
unit-tested with no server.

One rule runs through the layers above the structural one: **kind is the label, type is a property**
(``docs/GENERALIZATION.md``). The ``:Node`` layer still carries per-type labels because its
vocabulary is closed (``core.models.NodeType``); every semantic layer's type is induced per book, so
it rides as an indexed property instead of an unbounded label set.

- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle), plus an HTTPS
  Query-API transport for sandboxes where Bolt is blocked.
- ``nodes`` — the structural provenance layer: a ``:Source`` root per book with its ``:Node`` stream
  hanging off it (``:HEAD`` / ``:NEXT``), each node carrying its per-type label.
- ``entities`` — the ``:Entity:Mention`` overlay on that stream, rooted via ``:HAS_ENTITY`` and
  linked to its member chunks via ``:DERIVED_FROM``.
- ``procedures`` — the procedural half: a ``:Procedure`` per derivation (``:HAS_PROCEDURE``) rooting
  an ``:Event`` step chain (``:FIRST`` / ``:THEN``).
- ``concepts`` — the conceptualization axis: a global, born-canonical ``:Concept`` per induced tag,
  with ``:INSTANCE_OF`` edges from entities and from procedure steps.
- ``dependencies`` — concept-level prerequisites, ``(:Concept)-[:DEPENDS_ON]->(:Concept)``.
- ``references`` / ``uses`` / ``realizes`` — the citation layer: ``:REFERENCES`` edges onto global
  ``:Entity:Canonical`` hubs, the finer step-level ``:USES``, and the ``:REALIZES`` edge tying a hub
  back to the mention that defines it.
- ``schema`` — idempotent bootstrap (uuid constraints + the source/type/name lookup indexes).
- ``writer`` — one ``persist_*`` per layer, each a batched, idempotent MERGE on deterministic uuids.
- ``persister`` — the two pipeline stages that call them when Neo4j is configured.

Still to come (``docs/UNIFIED-KG.md``): the ``:DEMONSTRATES``/``:PRACTICES`` anchor edges and the
semantic dedup tier (embedding fusion) that refines ``:REALIZES`` and merges concept paraphrases.
"""
