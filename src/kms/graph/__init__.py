"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

Three semantic kinds over a provenance tier. Every mapping module is pure —
deterministic uuids, property maps, edge pairs — and free of the neo4j driver,
which is quarantined in ``db``.

- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle),
  plus an HTTPS Query-API transport for sandboxes where Bolt is blocked.
- ``nodes`` — ``models.ASTNode`` → ``:Node`` (+ its per-type label) and the
  ``:Source`` root. The provenance tier every semantic vertex points back at.
- ``statements`` — ``models.Statement`` → a bare ``:Statement``. Content
  is a single string written by the statement extractor.
- ``procedures`` — ``models.Procedure`` → ``:Procedure`` and its ``:Act`` step
  chain (declared but not yet written — the step decomposer is a future pass).
- ``schema`` — idempotent constraint/index bootstrap for every label.
- ``writer`` — the I/O half: ``persist_nodes`` / ``persist_statements`` /
  ``persist_procedures``, each batched and idempotent.
- ``persister`` — ``IngestionPersisterNode``, the terminal stage that persists
  everything in one pass.
"""
