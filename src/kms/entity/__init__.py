"""Phase 2 — the entity layer: build the sparse entity overlay on the flat node stream and enrich it.

Genre stages (domain-free, shared by every layer): the exercise ``splitter``, the
``instruction_finder``/``instruction_distributor`` pair, the ``collector`` fan-in, and — over the
collected overlay — the ``conceptualizer`` and ``dependency_finder``.

Extraction comes in two interchangeable layers between the node persister and the collector (see
``pipeline.build_graph``): the validated math path, three per-type chains under ``finders/`` and
``attributors/``; and the general path of ``docs/GENERALIZATION.md`` — ``finders/block`` (spans only)
→ ``attributors/universal`` (an induced open type) → ``procedure_finder``. Both end at the one
open-relation ``referencers/open`` pass.
"""
