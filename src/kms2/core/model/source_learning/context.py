"""Content-only context shared by source-learning inference passes."""

from pydantic import BaseModel, Field

from kms2.core.model.source_extraction import (
    SourceEvidenceInput,
)


class SourceLearningHubContext(BaseModel):
    """An original member statement used as shared learning context."""

    proposition_statement: str = Field(min_length=1)


class SourceLearningEvidence(BaseModel):
    """One proposition occurrence and its exact source evidence."""

    proposition_text: str = Field(min_length=1)
    source_context: SourceEvidenceInput
