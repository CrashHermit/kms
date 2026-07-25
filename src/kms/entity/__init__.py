"""Phase 2 — the block overlay: detect → attribute → decompose.

One sequential chain over the flat node stream (``docs/ARCHITECTURE.md``, rule 4). The pedagogy
stages run first — ``splitter`` makes each exercise its own node, ``instruction_finder`` tags
lead-ins — then ``group_finder`` walks the stream once and emits untyped ``entity`` and
``procedure`` spans, ``statement_extractor`` fills each block's attributes (including the open,
induced ``type``), ``procedure_extractor`` decomposes each derivation into verbatim steps, and
``instruction_distributor`` copies a lead-in's directive onto the blocks it governs.
"""
