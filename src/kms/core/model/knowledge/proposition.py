"""Canonical propositions and their persistent admission provenance."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from kms.core.model.base import Vertex


class Proposition(Vertex):
    """One self-contained canonical claim reusable as a premise."""

    content: str
    embedding: list[float]
    created_at: datetime


class Argument(Vertex):
    """One learner-facing argument joining premises to a conclusion."""

    explanation: str
    embedding: list[float]
    created_at: datetime


class KnowledgeAssertion(Vertex):
    """One turn's admission or reuse of a proposition and its evidence."""

    created_at: datetime


class KnowledgeAssertionRecord(BaseModel):
    """Loaded assertion plus its turn and direct evidence references."""

    model_config = ConfigDict(extra='forbid')

    assertion: KnowledgeAssertion
    proposition_uuid: str
    supporting_block_uuids: list[str]
    argument_uuids: list[str]
    teaching_turn_uuid: str


class PropositionRecord(BaseModel):
    """Canonical claim and bounded, one-hop provenance references."""

    model_config = ConfigDict(extra='forbid')

    proposition: Proposition
    assertions: list[KnowledgeAssertionRecord]
    argument_uuids: list[str]
