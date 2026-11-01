"""Content-only context shared by source-learning inference passes."""

from pydantic import BaseModel, Field

from kms2.core.model.source_extraction import SourceEvidenceInput


class SourceLearningEvidence(BaseModel):
    """One source-material unit and its exact source evidence."""

    source_text: str = Field(min_length=1)
    source_context: SourceEvidenceInput
