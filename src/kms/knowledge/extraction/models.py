"""Source-grounded knowledge creation inputs and candidates."""

from pydantic import BaseModel

from kms.knowledge.candidates import NewArgument, NewProposition
from kms.knowledge.models import Argument, Proposition
from kms.source.retrieval.models import SourceUnitMaterial


class UnitExtractionRequest(BaseModel):
    """An atomic understanding and its ordered source-unit evidence."""

    understanding: str
    context_before: list[SourceUnitMaterial]
    target: SourceUnitMaterial
    context_after: list[SourceUnitMaterial]
    existing_propositions: list[Proposition]
    existing_arguments: list[Argument]


class UnitExtractionCandidates(BaseModel):
    """Relevant source-grounded claims and argument candidates."""

    new_propositions: list[NewProposition]
    existing_proposition_uuids: list[str]
    new_arguments: list[NewArgument]
    existing_argument_uuids: list[str]


class UnitExtractionResult(BaseModel):
    """Candidates attributed to the source and target that produced them."""

    source_uuid: str
    target_unit_uuid: str
    candidates: UnitExtractionCandidates
