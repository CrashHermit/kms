"""LangGraph state for KMS2 source processing."""

from typing import TypedDict

from kms2.core.model import OCRArtifact, Source, SourceContent


class SourceState(TypedDict, total=False):
    """State shared by the nodes that process one source."""

    pdf_path: str
    pages: list[int] | None
    source: Source
    ocr_artifacts: list[OCRArtifact]
    content_corrector: list[SourceContent]
