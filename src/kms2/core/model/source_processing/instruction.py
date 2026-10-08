"""Instruction pointer model for KMS2."""

from pydantic import Field

from kms2.core.model.base import Vertex


class Instruction(Vertex):
    """An ordered block grouping with optional governed statements.

    Source ownership derives exclusively from nonempty, same-source block
    membership. Persistence retains MEMBER_OF and GOVERNS relationships,
    not a direct source ownership edge.
    """

    member_block_uuids: list[str] = Field(default_factory=list)
    governed_statement_uuids: list[str] = Field(default_factory=list)
