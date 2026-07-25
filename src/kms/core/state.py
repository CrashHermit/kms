"""
The LangGraph state that carries the pipeline's data structures through the graph.

This is the orchestration layer's channel schema — distinct from the domain models it
carries, which live in ``kms.core.models``. Parallel Send workers never mutate the
backbone directly: each worker returns a ``(segment_index, result)`` entry into a
per-stage reducer channel (operator.add, so concurrent writes merge instead of clashing),
and the stage's ``collect`` step drains that channel back into the matching models.Segment keyed
by its index (see ``models.merge_results_into_segments``). Because the merge keys into the
already-ordered backbone, document order is preserved without a separate sort.
"""

import operator
from typing import Annotated, TypedDict

from kms.core import models


class State(TypedDict, total=False):
    """Shared state for every stage.

    Two backbones, one after the other. During per-page ingestion (corrector, extractor,
    seam) `segments` is the ordered backbone. The seam merger then flattens the healed
    per-page nodes into `nodes` — the global ordered node list that the per-type entity
    finders each walk. `segments` is retained past that point only for its pictures
    (picture resolution at assembly). Both backbones use the default overwrite reducer
    because only the sequential collect steps write them.

    The `*_entities` channels are each written once, by their own finder — three of them
    (problem / definition / theorem, running in parallel) on the per-type entity layer, one
    (`block_entities`) on the block-finder layer; which set is live is a wiring choice, see
    `pipeline.build_graph`. They are sparse overlays and may reference the same node from more than
    one entity — that is fine, members are node-id pointers. Because the splitter has already made
    exercise nodes atomic (one node per exercise), a finder emits one entity per exercise with
    distinct members — no coarse-vs-fine reconciliation is needed.

    The collector stage is the fan-in: it flattens whichever overlays ran into `entities`, the one
    document-ordered, globally-id'd list every stage after it (conceptualizer, dependency finder,
    entity persister) reads. `concept_dependencies` is the dependency finder's concept-level
    prerequisite graph, the one channel whose unit is a concept rather than an entity.

    The `*_results` channels are map-reduce scratch space: parallel Send workers append
    entries and the stage's collect step drains them back into the active backbone. They
    carry an operator.add reducer so concurrent worker writes merge rather than clash.
    """

    segments: list[models.Segment]
    nodes: list[models.ASTNode]
    source: str  # book identity (the graph persister's Neo4j key); set by run()
    source_metadata: dict[
        str, str
    ]  # book attributes (title, author, …) for the :Source node
    problem_entities: list[models.Entity]  # written by the problem finder
    definition_entities: list[models.Entity]  # written by the definition finder
    theorem_entities: list[models.Entity]  # written by the theorem finder
    block_entities: list[models.Entity]  # written by the block finder
    entities: list[models.Entity]  # the flattened overlay (collector fan-in)
    concept_dependencies: list[
        models.Dependency
    ]  # concept-level prerequisites (dependency finder)
    correction_results: Annotated[
        list[tuple[int, str]], operator.add
    ]  # (segment index, corrected markdown)
    extract_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
    seam_even_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
    seam_odd_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
