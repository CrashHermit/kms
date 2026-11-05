"""Owned proposition search results and bounded graph read models."""

from pydantic import BaseModel, ConfigDict

from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import Proposition


class PropositionSearchCandidate(BaseModel):
    """User-owned proposition scored by cosine similarity."""

    model_config = ConfigDict(extra='forbid')

    proposition: Proposition
    similarity: float


class PropositionSearchHit(PropositionSearchCandidate):
    """Search candidate with centralized reranker relevance."""

    relevance_score: float


class ArgumentSearchCandidate(BaseModel):
    """Complete user-owned argument scored by vector similarity."""

    model_config = ConfigDict(extra='forbid')

    record: ArgumentRecord
    content: str
    similarity: float


class ArgumentSearchHit(ArgumentSearchCandidate):
    """Argument search candidate with centralized reranker relevance."""

    relevance_score: float
