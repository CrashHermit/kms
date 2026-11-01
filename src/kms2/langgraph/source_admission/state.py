"""Invocation state for researched source snapshot admission."""

from pydantic import BaseModel, Field

from kms2.core.model.page import SourcePage
from kms2.core.model.source_processing.admission import (
    ResearchSourceMaterial,
    SourceAdmissionResult,
    SourceSnapshotPreparation,
)


class SourceAdmissionState(BaseModel):
    """Carry one capture through preparation, processing, and finalization."""

    user_uuid: str
    material: ResearchSourceMaterial
    preparation: SourceSnapshotPreparation | None = None
    final_snapshot: SourceAdmissionResult | None = None
    processed_pages: list[SourcePage] = Field(default_factory=list)
