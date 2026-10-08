"""Backend and model-facing contracts for coherent flashcard generation."""

from pydantic import BaseModel, Field

from kms2.core.model.source_learning.atomic import SourceAtomicFlashcardRequest
from kms2.core.model.source_learning.context import (
    SourceLearningEvidence,
    SourceLearningHubContext,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcardCandidate,
    SourceFlashcardOccurrence,
)


class SourceAtomicFlashcardPacket(BaseModel):
    """Retained atomic request and cards for one coherent batch item."""

    request: SourceAtomicFlashcardRequest
    cards: list[SourceFlashcardOccurrence] = Field(default_factory=list)


class SourceIndexedFlashcard(BaseModel):
    """Content-only first-pass card with a request-local index."""

    index: int = Field(ge=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class SourceCoherentFlashcardPacketInput(BaseModel):
    """One evidence packet and its indexed first-pass cards."""

    evidence: SourceLearningEvidence
    cards: list[SourceIndexedFlashcard] = Field(min_length=1)


class SourceCoherentFlashcardInput(BaseModel):
    """Content-only input for one same-hub coherent generation call."""

    context: SourceLearningHubContext
    packets: list[SourceCoherentFlashcardPacketInput] = Field(min_length=1)


class SourceCoherentFlashcardRequest(BaseModel):
    """Backend request retaining atomic packets for one coherent call."""

    hub_uuid: str = Field(min_length=1)
    context: SourceLearningHubContext
    packets: list[SourceAtomicFlashcardPacket] = Field(min_length=1)

    def model_input(self) -> SourceCoherentFlashcardInput:
        """Project packets to content with contiguous local card indexes."""
        index = 1
        packet_inputs: list[SourceCoherentFlashcardPacketInput] = []
        for packet in self.packets:
            indexed_cards: list[SourceIndexedFlashcard] = []
            for occurrence in packet.cards:
                indexed_cards.append(
                    SourceIndexedFlashcard(
                        index=index,
                        question=occurrence.card.question,
                        answer=occurrence.card.answer,
                    )
                )
                index += 1
            packet_inputs.append(
                SourceCoherentFlashcardPacketInput(
                    evidence=packet.request.input.evidence,
                    cards=indexed_cards,
                )
            )
        return SourceCoherentFlashcardInput(
            context=self.context,
            packets=packet_inputs,
        )


class SourceCoherentFlashcardCandidate(SourceFlashcardCandidate):
    """Content-only coherent card with request-local parent indexes."""

    card_indexes: list[int] = Field(min_length=2)


__all__ = [
    'SourceAtomicFlashcardPacket',
    'SourceCoherentFlashcardCandidate',
    'SourceCoherentFlashcardInput',
    'SourceCoherentFlashcardPacketInput',
    'SourceCoherentFlashcardRequest',
    'SourceIndexedFlashcard',
]
