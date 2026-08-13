import operator
from typing import Annotated, TypedDict

from kms.core import models


class State(TypedDict, total=False):
    segments: list[models.Segment]
    nodes: list[models.ASTNode]
    source: str
    source_metadata: dict[str, str]
    spans: list[list[int]]
    instructions: list[models.Instruction]
    statements: list[models.Statement]
    procedures: list[models.Procedure]
    triplets: list[models.Triplet]
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
