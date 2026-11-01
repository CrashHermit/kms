"""Owned proposition search results and bounded graph read models."""

from pydantic import BaseModel, ConfigDict

from kms2.core.model.knowledge.proposition import Proposition


class PropositionSearchCandidate(BaseModel):
    """User-owned proposition scored by cosine similarity."""

    model_config = ConfigDict(extra='forbid')

    proposition: Proposition
    similarity: float


class PropositionSearchHit(PropositionSearchCandidate):
    """Search candidate with centralized reranker relevance."""

    relevance_score: float
