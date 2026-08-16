"""Shared langgraph state schema for the ingestion pipeline."""

import operator
from typing import Annotated, TypedDict

from kms.core import models


class State(TypedDict, total=False):
    """The mutable state threaded through every graph node.

    All keys are optional; worker results are merged in via the
    Annotated reducer fields. ``source`` and ``source_metadata`` name
    the document being ingested, ``segments`` and ``nodes`` carry the
    parsed document, and the remaining keys accumulate the extracted
    knowledge before persistence.
    """

    pdf_path: str
    output_dir: str
    pages: list[int] | None
    segments: list[models.Segment]
    nodes: list[models.ASTNode]
    source: str
    source_metadata: dict[str, str]
    spans: list[list[int]]
    instructions: list[models.Instruction]
    statements: list[models.Statement]
    procedures: list[models.Procedure]
    triplets: list[models.Triplet]
    entity_descriptions: dict[int, dict[str, str | None]]
    predicate_descriptions: dict[int, dict[str, str | None]]
    entity_embeddings: dict[int, dict[str, list[float]]]
    predicate_embeddings: dict[int, dict[str, list[float]]]
    entity_assigned: int
    predicate_assigned: int
    entity_hubs_created: int
    predicate_hubs_created: int
    entity_name_hubs_created: int
    predicate_name_hubs_created: int
    triplet_hubs_created: int
    procedures_created: int
    correction_results: Annotated[list[tuple[int, str]], operator.add]
    format_results: Annotated[list[tuple[int, str]], operator.add]
    extract_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
    seam_even_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
    seam_odd_results: Annotated[
        list[tuple[int, list[models.ASTNode]]], operator.add
    ]
