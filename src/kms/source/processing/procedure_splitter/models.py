"""Boundary models for splitting source procedure passages."""

from pydantic import BaseModel, Field

from kms.source.context import SourceBlockContext, SourceBlockWindow


class ProcedureSplitRequest(BaseModel):
    flat_position: int
    window: SourceBlockWindow


class ProcedureSplitRouteDecision(BaseModel):
    position: int = Field(ge=0, strict=True)
    requires_split: bool


class ProcedureSplitCandidate(BaseModel):
    position: int
    source_block: SourceBlockContext


class ProcedureSplitPiece(BaseModel):
    content: str


class ProcedureSplitResult(BaseModel):
    flat_position: int
    pieces: list[ProcedureSplitPiece] | None = None


class ProcedureSplitDecision(BaseModel):
    position: int
    pieces: list[ProcedureSplitPiece]
