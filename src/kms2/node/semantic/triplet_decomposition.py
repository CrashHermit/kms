"""LangGraph workers for source-faithful triplet decomposition."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import (
    SourceEntity,
    SourceEvent,
    SourceFact,
    SourcePredicate,
    SourceTriplet,
    SourceTripletOccurrence,
    TripletDecompositionRequest,
    TripletDecompositionResult,
    TripletEndpointKind,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.triplet_decomposition import TripletDecomposerModule


class TripletWorkerState(TypedDict):
    """State supplied to one triplet-decomposition worker."""

    triplet_decomposition_request: TripletDecompositionRequest


class TripletDecompositionNode:
    """Dispatch facts, collect candidates, and create triplet occurrences."""

    def __init__(self, decomposer: TripletDecomposerModule) -> None:
        self._decomposer = decomposer

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['triplet_collect']:
        """Dispatch one decomposition request per extracted fact."""
        sends = [
            Send(
                'triplet_worker',
                {
                    'triplet_decomposition_request': TripletDecompositionRequest(
                        fact_index=index,
                        fact=fact,
                    )
                },
            )
            for index, fact in enumerate(state.extracted_facts)
        ]
        return sends or 'triplet_collect'

    async def worker(
        self, state: TripletWorkerState
    ) -> dict[str, list[TripletDecompositionResult]]:
        """Decompose one source fact."""
        request = state['triplet_decomposition_request']
        triplets = await self._decomposer.aforward(request=request)
        return {
            'triplet_results': [
                TripletDecompositionResult(
                    fact_index=request.fact_index,
                    fact=request.fact,
                    triplets=triplets,
                )
            ]
        }

    def collect(
        self, state: SemanticState
    ) -> dict[str, list[SourceFact] | list[SourceTripletOccurrence]]:
        """Materialize each candidate as a fresh source triplet occurrence."""
        results = sorted(
            state.triplet_results,
            key=lambda item: item.fact_index,
        )
        source_facts: list[SourceFact] = []
        triplet_occurrences: list[SourceTripletOccurrence] = []
        for result in results:
            fact = result.fact
            source_fact = SourceFact(
                source_uuid=fact.source_uuid,
                source_block_uuid=fact.source_block_uuid,
                text=fact.text,
            )
            source_facts.append(source_fact)
            for candidate in result.triplets:
                subject = (
                    SourceEntity(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.subject,
                    )
                    if candidate.subject_kind is TripletEndpointKind.ENTITY
                    else SourceEvent(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.subject,
                    )
                )
                object_ = (
                    SourceEntity(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.object,
                    )
                    if candidate.object_kind is TripletEndpointKind.ENTITY
                    else SourceEvent(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.object,
                    )
                )
                predicate = SourcePredicate(
                    source_uuid=fact.source_uuid,
                    source_block_uuid=fact.source_block_uuid,
                    predicate=candidate.predicate,
                )
                triplet = SourceTriplet(
                    source_uuid=fact.source_uuid,
                    source_block_uuid=fact.source_block_uuid,
                    subject_uuid=subject.uuid,
                    object_uuid=object_.uuid,
                    predicate_uuid=predicate.uuid,
                )
                triplet_occurrences.append(
                    SourceTripletOccurrence(
                        fact=source_fact,
                        triplet=triplet,
                        subject=subject,
                        object=object_,
                        predicate=predicate,
                    )
                )
        return {
            'source_facts': source_facts,
            'triplet_occurrences': triplet_occurrences,
        }


__all__ = ['TripletDecompositionNode', 'TripletWorkerState']
