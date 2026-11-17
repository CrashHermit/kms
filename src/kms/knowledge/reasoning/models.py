"""Demand-driven proposition reasoning inputs and candidates."""

from pydantic import BaseModel

from kms.knowledge.candidates import NewArgument, NewProposition
from kms.knowledge.models import KnowledgeContext


class ReasoningRequest(BaseModel):
    """An atomic understanding and the complete knowledge context offered."""

    understanding: str
    context: KnowledgeContext


class ReasoningCandidates(BaseModel):
    """New conclusions and supporting argument candidates."""

    new_propositions: list[NewProposition]
    new_arguments: list[NewArgument]
