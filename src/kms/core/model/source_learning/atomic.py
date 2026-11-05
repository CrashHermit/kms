"""Backend and model-facing contracts for atomic flashcard generation."""

from pydantic import BaseModel, Field

from kms.core.content import MultimodalContentBuilder
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source_extraction import SourceEvidenceInput
from kms.core.model.source_learning.context import SourceLearningEvidence
from kms.core.model.source_learning.flashcard import SourceFlashcardCandidate
from kms.core.source_context import project_source_window


class SourceAtomicFlashcardInput(BaseModel):
    """Content supplied to one atomic flashcard inference request."""

    evidence: SourceLearningEvidence


class SourceAtomicFlashcardMaterial(BaseModel):
    """Ordered metadata identifying one eligible source owner."""

    source_block_uuid: str = Field(min_length=1)
    source_page_index: int = Field(ge=0)


class SourceAtomicFlashcardRequest(SourceAtomicFlashcardMaterial):
    """Backend request retaining canonical evidence for one inference call."""

    window: SourceBlockWindow

    def model_input(
        self, builder: MultimodalContentBuilder
    ) -> SourceAtomicFlashcardInput:
        """Project this request to a call-local native-image catalog."""
        projected = project_source_window(self.window, builder)
        target = self.window.target[0]
        return SourceAtomicFlashcardInput(
            evidence=SourceLearningEvidence(
                source_text=target.content,
                source_context=SourceEvidenceInput(
                    context_before=projected.context_before,
                    target_blocks=projected.target,
                    context_after=projected.context_after,
                ),
            )
        )


__all__ = [
    'SourceAtomicFlashcardInput',
    'SourceAtomicFlashcardMaterial',
    'SourceAtomicFlashcardRequest',
    'SourceFlashcardCandidate',
]
