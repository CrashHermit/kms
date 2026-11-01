"""Invocation-only state for proposition extraction and identity preparation."""

from pydantic import BaseModel, ConfigDict

from kms2.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeDraft,
    KnowledgeExtractionInput,
)
from kms2.core.model.source import SourceRevisionReference


class KnowledgeState(BaseModel):
    """Private backend tables and outputs for one admission preparation."""

    model_config = ConfigDict(extra='forbid')

    user_uuid: str
    request: KnowledgeExtractionInput
    existing_proposition_uuids: list[str]
    supporting_block_uuids: list[str]
    block_source_revisions: list[SourceRevisionReference]
    evidence_image_bytes: list[bytes]
    expected_revision: int | None = None
    draft: KnowledgeDraft | None = None
    admission: KnowledgeAdmission | None = None
