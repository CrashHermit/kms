"""Propositions and arguments in the knowledge graph."""

from pydantic import BaseModel

from kms.core.models import Vertex


class Proposition(Vertex):
    """An identifiable claim, with a short name and complete statement."""

    name: str
    statement: str


class Argument(Vertex):
    """Propositions that jointly support a conclusion.

    Premise and conclusion UUIDs identify propositions without embedding their
    statements. Persist these associations as graph relationships.
    """

    premise_uuids: list[str]
    conclusion_uuid: str


class KnowledgeContext(BaseModel):
    """Matched claims and a closed set of connected reasoning."""

    matched: list[Proposition]
    propositions: list[Proposition]
    arguments: list[Argument]


class StoredKnowledge(BaseModel):
    """Graph objects stored or reused for one candidate result."""

    propositions: list[Proposition]
    arguments: list[Argument]


class ReasonedKnowledge(BaseModel):
    """Retrieved context and resulting constructed reasoning."""

    context: KnowledgeContext
    stored: StoredKnowledge
