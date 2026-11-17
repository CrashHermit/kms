"""Knowledge sufficiency judgment contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.knowledge.models import KnowledgeContext


class SufficiencyRequest(BaseModel):
    """An atomic understanding and retrieved knowledge to assess."""

    model_config = ConfigDict(extra='forbid')

    understanding: str
    context: KnowledgeContext


class SufficientKnowledge(BaseModel):
    """The supplied knowledge addresses the requested understanding."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['sufficient'] = 'sufficient'
    explanation: str


class InsufficientKnowledge(BaseModel):
    """The supplied knowledge does not address the requested understanding."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['insufficient'] = 'insufficient'
    explanation: str
    missing_knowledge: str


SufficiencyJudgment = Annotated[
    SufficientKnowledge | InsufficientKnowledge, Field(discriminator='kind')
]
