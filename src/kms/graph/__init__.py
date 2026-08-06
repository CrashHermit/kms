"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

Three semantic kinds over a provenance tier. Every mapping module is pure —
deterministic uuids, property maps, edge pairs — and free of the neo4j driver,
which is quarantined in ``db``.

- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle).
- ``nodes`` — ``models.ASTNode`` → ``:Node`` (+ its per-type label) and the
  ``:Source`` root. The provenance tier every semantic vertex points back at.
- ``statements`` — ``models.Statement`` → a bare ``:Statement`` hub.
  Statements carry no text; their members' raw nodes carry it.
- ``procedures`` — ``models.Procedure`` → ``:Procedure`` and its ``:Act`` step
  chain (declared but not yet written — the step decomposer is a future pass).
- ``entities`` — per-triplet ``:Entity`` vertices: one per (node, triplet,
  role), carrying the enricher's description. Each triplet's subject and
  object are their own vertices — zero sharing between triplets. Reached
  through their triplets (``:HAS_SUBJECT``/``:HAS_OBJECT``), with no direct
  ``:Node`` anchor edge.
- ``triplets`` — ``models.Triplet`` → ``:Triplet`` hubs: the verbatim
  assertion (subject/predicate/object strings), written per node occurrence,
  hung off its ``:Fact`` via ``:YIELDS`` with ``:HAS_SUBJECT``/``:HAS_OBJECT``
  endpoints.
- ``predicates`` — the described predicate component: one ``:Predicate`` per
  (node, triplet) carrying the predicate text + description, reached from its
  triplet hub via ``:HAS_PREDICATE``.
- ``schema`` — idempotent constraint/index bootstrap for every label.
- ``queries`` — every Cypher statement in one place: the batched MERGE
  upserts the writer runs plus named read lookbacks (distinct relation
  types for the living schema) returning plain data over the injected
  session factory.
- ``writer`` — the I/O half: composes rows and edge pairs and hands them
  to the queries in ``queries``; each write is batched and idempotent.
- ``persister`` — ``IngestionPersisterNode``, the terminal stage that persists
  everything in one pass.
"""
