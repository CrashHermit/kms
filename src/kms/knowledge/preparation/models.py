"""Contracts for preparing knowledge about one understanding."""

from pydantic import BaseModel

from kms.knowledge.models import KnowledgeContext
from kms.knowledge.sufficiency.models import SufficiencyJudgment


class PreparationResult(BaseModel):
    """Prepared knowledge and the final sufficiency judgment."""

    understanding: str
    context: KnowledgeContext
    judgment: SufficiencyJudgment
