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
- ``equations`` — ``models.Equation`` → ``:Equation`` hung off its
  provenance ``:Node`` via ``:HAS_EQUATION``; hubs inherit through
  ``:MEMBER_OF``.
- ``variables`` — ``models.Variable`` → ``:Variable`` hung off its ``:Node``
  or ``:Equation`` via ``:HAS_VARIABLE``; hubs inherit through
  ``:MEMBER_OF``.
- ``schema`` — idempotent constraint/index bootstrap for every label.
- ``queries`` — the read side of the I/O layer: named, parameterised
  lookbacks (existing concepts for dedup, distinct relation types for the
  living schema) returning plain data over the injected session factory.
- ``writer`` — the I/O half: ``persist_nodes`` / ``persist_statements`` /
  ``persist_procedures``, each batched and idempotent.
- ``persister`` — ``IngestionPersisterNode``, the terminal stage that persists
  everything in one pass.
"""
