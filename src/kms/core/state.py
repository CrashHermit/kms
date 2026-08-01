"""
The LangGraph state that carries the pipeline's data structures through the
graph.

This is the orchestration layer's channel schema — distinct from the domain
models it carries, which live in ``kms.core.models``. Parallel Send workers
never mutate the backbone directly: each worker returns a
``(segment_index, result)`` entry into a per-stage reducer channel
(operator.add, so concurrent writes merge instead of clashing), and the stage's
``collect`` step drains that channel back into the matching models.Segment
keyed by its index (see ``models.merge_results_into_segments``). Because the
merge keys into the already-ordered backbone, document order is preserved
without a separate sort.
"""

import operator
from typing import Annotated, TypedDict

from kms.core import models


class State(TypedDict, total=False):
    """Shared state for every stage.

    Two backbones, one after the other. During per-page ingestion (corrector,
    extractor, seam) `segments` is the ordered backbone. The seam merger then
    flattens the healed per-page nodes into `nodes` — the global ordered node
    list that the group finder walks. `segments` is retained past that point
    only for its pictures (picture resolution at assembly). Both backbones use
    the default overwrite reducer because only the sequential collect steps
    write them.

    `spans` carries the PCF's compound unit spans (member node ids, document
    order) to the role typer, which diagnoses each group's composition. The PCF
    groups related content together; it does not split statements from
    procedures.

    `statements` carries the statement overlay from the role typer to the
    persister: one Statement HUB per group, each holding its block and its
    members' node ids. It is a channel of its own, deliberately NOT part of
    `nodes`: the hub's members are already in the stream, so placing the hub
    there would represent every member twice — once as itself and once inside
    its statement.

    `procedures` carries the procedure overlay from the role typer to the
    partitioners and the persister: one Procedure HUB per derivation block,
    holding its members' node ids. Like `statements`, it is its own channel,
    never part of `nodes`.

    The `*_results` channels are map-reduce scratch space: parallel Send
    workers append entries and the stage's collect step drains them back into
    the active backbone. They carry an operator.add reducer so concurrent
    worker writes merge rather than clash.
    """

    segments: list[models.Segment]
    nodes: list[models.ASTNode]
    source: str  # book identity (the graph persister's Neo4j key); set by run()
    source_metadata: dict[
        str, str
    ]  # book attributes (title, author, …) for the :Source node
    spans: list[
        list[int]
    ]  # untyped unit spans from the group finder, document order
    statements: list[
        models.Statement
    ]  # the statement overlay, group order, from the role typer
    procedures: list[models.Procedure]  # the procedure overlay, group order
    variables: Annotated[
        list[tuple[str, list[int], list[models.Variable]]], operator.add
    ]  # (unit_kind, block, variable bindings) from the variable extractor
    equations: Annotated[
        list[tuple[str, list[int], list[models.Equation]]], operator.add
    ]  # (unit_kind, block, equations) extracted per unit
    correction_results: Annotated[
        list[tuple[int, str]], operator.add
    ]  # (segment index, corrected markdown)
    format_results: Annotated[
        list[tuple[int, str]], operator.add
    ]  # (segment index, formatted markdown)
    extract_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
    seam_even_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
    seam_odd_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
