"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

So far:
- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle).
- ``nodes`` — the graph representation of the structural node stream (the ``:Node`` layer,
  reusing ``core.NodeType``): a pure ASTNode→Neo4j property mapping + deterministic uuid identity.
- ``schema`` — idempotent bootstrap for that layer (uuid constraint + type index).

The structural nodes are the provenance layer. The semantic tiers above them (dedup canonicals,
general entities, concepts) are still being designed and not modelled yet.
"""
