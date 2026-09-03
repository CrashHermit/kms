"""LangGraph OCR node for KMS2."""

from dataclasses import dataclass

from kms2.core.model import OCRArtifact
from kms2.langgraph.source.state import SourceState
from kms2.ocr.provider import OCRProvider


@dataclass
class OCRNode:
    """Delegate source OCR to an injected provider."""

    provider: OCRProvider

    def run(self, state: SourceState) -> dict[str, list[OCRArtifact]]:
        """Extract artifacts for the PDF represented by graph state."""
        artifacts = self.provider.extract(
            state['pdf_path'],
            pages=state.get('pages'),
        )
        return {'ocr_artifacts': artifacts}
