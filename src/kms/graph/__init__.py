"""Phase 3 — graph tier: the knowledge graph itself (Neo4j).

Foundation so far is just the plumbing: ``db`` — the quarantined async Neo4j driver
(connection, config, lifecycle). No graph schema or node/edge model is committed yet; the
tiers (dedup canonicals, general entities, concepts) are still being designed, and the only
node vocabulary we're sure of is the structural markdown types, which already live in
``core.models`` (``NodeType``). Modelling and schema bootstrap come once that's settled.
"""
