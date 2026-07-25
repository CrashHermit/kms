"""Phase 3 — graph tier: the knowledge graph itself (Neo4j). See ``docs/SCHEMA.md``.

Four semantic kinds over a provenance tier. Every mapping module is pure — deterministic uuids,
property maps, edge pairs — and free of the neo4j driver, which is quarantined in ``db``.

- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle), plus an HTTPS
  Query-API transport for sandboxes where Bolt is blocked.
- ``nodes`` — ``models.ASTNode`` → ``:Node`` (+ its per-type label) and the ``:Source`` root.
  The provenance tier every semantic vertex points back at.
- ``entities`` — ``models.Entity`` → a bare ``:Entity``. No per-type label: ``type`` is an open,
  induced property, and an open vocabulary would explode the label set.
- ``procedures`` — ``models.Procedure`` → ``:Procedure`` and its ``:Act`` step chain. Both bare:
  proof/solution is derivable from the owning entity, and an act's role was a closed taxonomy
  nothing read.
- ``concepts`` — the global ``:Concept`` hub identity scheme. **Currently dark**: nothing writes
  concepts or ``:INSTANCE_OF`` edges (see ``docs/CONCEPT-LAYER.md``).
- ``schema`` — idempotent constraint/index bootstrap for every label.
- ``writer`` — the I/O half: ``persist_nodes`` / ``persist_entities`` / ``persist_procedures``,
  each batched and idempotent on the deterministic uuids.
- ``persister`` — the two pipeline stages, ``NodePersisterNode`` and ``EntityPersisterNode``.

Deleted with the AutoMathKG rip-out (``docs/REBUILD.md``): the ``:Mention``/``:Canonical`` role
split and the ``references``/``uses``/``realizes`` modules that minted it.
"""
