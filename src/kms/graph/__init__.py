"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

So far — the structural node (``:Node``) provenance layer, end to end:
- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle).
- ``nodes`` — pure ASTNode→Neo4j mapping (reusing ``core.NodeType``): deterministic uuid identity,
  property map, and the per-type label (nodes carry ``:Node`` + ``:Math``/``:Paragraph``/…).
- ``schema`` — idempotent bootstrap (the uuid uniqueness constraint on ``:Node``).
- ``writer`` — ``persist_nodes``: batched multi-label MERGE + ``:NEXT`` reading-order edges,
  idempotent on the deterministic uuid.

The semantic tiers above this layer (dedup canonicals, general entities, concepts) are still
being designed and not modelled yet. Wiring ``persist_nodes`` into the pipeline (where ``source``
comes from) is the next step.
"""
