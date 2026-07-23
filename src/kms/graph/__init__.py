"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

So far — the structural provenance layer, end to end: a ``:Source`` root per book with its
``:Node`` stream hanging off it.
- ``db`` — the quarantined async Neo4j driver (connection, config, lifecycle).
- ``nodes`` — pure ASTNode→Neo4j mapping (reusing ``core.NodeType``): deterministic uuid identity
  for nodes and sources, property maps, and the per-type label (nodes carry ``:Node`` +
  ``:Math``/``:Paragraph``/…). Each node links back to its ``:Source`` via a ``source`` property.
- ``schema`` — idempotent bootstrap (uuid constraints on ``:Node``/``:Source`` + a ``source`` index).
- ``writer`` — ``persist_nodes``: MERGE the ``:Source`` root and the batched multi-label nodes, then
  wire ``(:Source)-[:HEAD]->`` first node and the ``:NEXT`` chain. Idempotent on deterministic uuids.

The semantic tiers above this layer (dedup canonicals, general entities, concepts) are still
being designed and not modelled yet. Wiring ``persist_nodes`` into the pipeline (where ``source``
and its metadata come from) is the next step.
"""
