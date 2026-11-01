"""Initialize source-processing pages from a typed input."""

from kms2.core.model.page import SourcePage
from kms2.core.model.source_processing.input import (
    OCRSourceInput,
    TextSourceInput,
)
from kms2.langgraph.source_processing.state import SourceProcessingState


class SourceInputNode:
    """Route inputs and hydrate prepared pages without rerunning OCR."""

    def route(self, state: SourceProcessingState) -> str:
        """Return the typed input kind for initial graph routing."""
        return state.input.kind

    def load_ocr(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Map prepared provider-neutral OCR artifacts to canonical pages."""
        source_input = state.input
        assert isinstance(source_input, OCRSourceInput)
        return {'ocr_pages': source_input.pages}

    def load_text(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Load prepared text directly into the formatting input."""
        source_input = state.input
        assert isinstance(source_input, TextSourceInput)
        return {'corrected_pages': source_input.pages}
