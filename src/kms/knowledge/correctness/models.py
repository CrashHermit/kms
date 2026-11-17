"""Typed requests and judgments for source-grounded correctness."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.knowledge.models import Argument, Proposition
from kms.source.retrieval.models import SourceUnitWindow


class PropositionToCheck(BaseModel):
    """One proposition to compare with source evidence."""

    kind: Literal['proposition'] = 'proposition'
    proposition: Proposition


class ArgumentToCheck(BaseModel):
    """An argument and its premise and conclusion propositions."""

    kind: Literal['argument'] = 'argument'
    argument: Argument
    propositions: list[Proposition]


class CorrectnessRequest(BaseModel):
    """One knowledge item and the source windows supplied as evidence."""

    model_config = ConfigDict(extra='forbid')

    item: Annotated[
        PropositionToCheck | ArgumentToCheck, Field(discriminator='kind')
    ]
    evidence: list[SourceUnitWindow]


class CorrectnessJudgment(BaseModel):
    """A verdict on the item relative to the supplied evidence."""

    model_config = ConfigDict(extra='forbid')

    verdict: Literal['supported', 'contradicted', 'insufficient_evidence']
    explanation: str
