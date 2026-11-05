"""Evidence shared by the two source-learning inference passes."""

from pydantic import BaseModel

from kms.core.model.source_extraction import SourceEvidenceInput


class SourceLearningEvidence(BaseModel):
    """One source-material unit and its exact source evidence."""

    source_text: str | None
    source_context: SourceEvidenceInput
