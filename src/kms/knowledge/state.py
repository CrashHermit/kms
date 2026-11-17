"""Shared state for sequential knowledge workflows."""

from typing import TypedDict

from kms.knowledge.extraction.models import UnitExtractionResult
from kms.knowledge.models import KnowledgeContext, StoredKnowledge
from kms.knowledge.reasoning.models import ReasoningCandidates
from kms.knowledge.sufficiency.models import SufficiencyJudgment
from kms.source.retrieval.models import SourceUnitSearchHit


class KnowledgeState(TypedDict, total=False):
    user_uuid: str
    understanding: str
    context: KnowledgeContext
    retrieved_context: KnowledgeContext
    unit_uuid: str
    pending: UnitExtractionResult | ReasoningCandidates | None
    stored: StoredKnowledge
    judgment: SufficiencyJudgment
    source_hits: list[SourceUnitSearchHit]
    source_index: int
    has_reasoned: bool
    has_searched_sources: bool
