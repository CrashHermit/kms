"""
The LangGraph state that carries the pipeline's data structures through the graph.

This is the orchestration layer's channel schema — distinct from the domain models it
carries, which live in ``kms.core.models``. Parallel Send workers never mutate the
backbone directly: each worker returns a ``(segment_index, result)`` entry into a
per-stage reducer channel (operator.add, so concurrent writes merge instead of clashing),
and the stage's ``collect`` step drains that channel back into the matching Segment keyed
by its index (see ``merge_results_into_segments``). Because the merge keys into the
already-ordered backbone, document order is preserved without a separate sort.
"""

import operator
from typing import Annotated, TypedDict

from kms.core.models import ASTNode, Entity, Segment


class State(TypedDict, total=False):
    """Shared state for every stage.

    Two backbones, one after the other. During per-page ingestion (corrector, extractor,
    seam) `segments` is the ordered backbone. The seam merger then flattens the healed
    per-page nodes into `nodes` — the global ordered node list that the per-type entity
    finders each walk. `segments` is retained past that point only for its pictures
    (picture resolution at assembly). Both backbones use the default overwrite reducer
    because only the sequential collect steps write them.

    The three `*_entities` channels are each written once, by their own finder (the
    finders run in parallel over `nodes`). They are independent sparse overlays and may
    reference the same node from more than one entity — that is fine, members are node-id
    pointers. Because the splitter has already made exercise nodes atomic (one node per
    exercise), the problem finder emits one entity per exercise with distinct members — no
    coarse-vs-fine reconciliation is needed. `run()` concatenates the three into one flat,
    document-ordered entity list with global ids for the emitted `entities.json`.

    The `*_results` channels are map-reduce scratch space: parallel Send workers append
    entries and the stage's collect step drains them back into the active backbone. They
    carry an operator.add reducer so concurrent worker writes merge rather than clash.
    """

    segments: list[Segment]
    nodes: list[ASTNode]
    source: str  # book identity (the graph persister's Neo4j key); set by run()
    problem_entities: list[Entity]  # written by the problem finder
    definition_entities: list[Entity]  # written by the definition finder
    theorem_entities: list[Entity]  # written by the theorem finder
    correction_results: Annotated[
        list[tuple[int, str]], operator.add
    ]  # (segment index, corrected markdown)
    extract_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_even_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
    seam_odd_results: Annotated[list[tuple[int, list[ASTNode]]], operator.add]
