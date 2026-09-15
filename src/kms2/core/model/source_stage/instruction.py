"""Instruction pointer model for KMS2."""

from pydantic import Field

from kms2.core.model.base import Vertex


class Instruction(Vertex):
    """An ordered pointer to instruction members and governed statements."""

    member_block_uuids: list[str] = Field(default_factory=list)
    governed_statement_uuids: list[str] = Field(default_factory=list)
