"""LangGraph workers for source-faithful triplet decomposition."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_semantic.source_entity import SourceEntity
from kms2.core.model.source_semantic.source_event import SourceEvent
from kms2.core.model.source_semantic.source_predicate import SourcePredicate
from kms2.core.model.source_semantic.source_triplet import (
    SourceTriplet,
    SourceTripletOccurrence,
)
from kms2.core.model.source_semantic.source_triplet_decomposition import (
    SourceTripletDecompositionRequest,
    SourceTripletDecompositionResult,
    SourceTripletEndpointKind,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_triplet_decomposition import (
    SourceTripletDecomposerModule,
)


class SourceTripletWorkerState(TypedDict):
    """State supplied to one triplet-decomposition worker."""

    triplet_decomposition_request: SourceTripletDecompositionRequest


class SourceTripletDecompositionNode:
    """Dispatch persisted facts and collect typed triplet occurrences."""

    def __init__(self, decomposer: SourceTripletDecomposerModule) -> None:
        self._decomposer = decomposer

    def dispatch(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_triplet_collect']:
        """Dispatch one decomposition request per persisted source fact."""
        sends = [
            Send(
                'source_triplet_worker',
                {
                    'triplet_decomposition_request': SourceTripletDecompositionRequest(
                        fact=fact
                    )
                },
            )
            for fact in state.source_facts
        ]
        return sends or 'source_triplet_collect'

    async def worker(
        self, state: SourceTripletWorkerState
    ) -> dict[str, list[SourceTripletDecompositionResult]]:
        """Decompose one persisted source fact."""
        request = state['triplet_decomposition_request']
        triplets = await self._decomposer.aforward(request=request)
        return {
            'triplet_results': [
                SourceTripletDecompositionResult(
                    fact=request.fact,
                    triplets=triplets,
                )
            ]
        }

    def collect(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourceTripletOccurrence]]:
        """Materialize candidates without recreating their source facts."""
        positions = {
            fact.uuid: index for index, fact in enumerate(state.source_facts)
        }
        results = sorted(
            state.triplet_results,
            key=lambda item: positions[item.fact.uuid],
        )
        triplet_occurrences: list[SourceTripletOccurrence] = []
        for result in results:
            fact = result.fact
            source_block_uuid = fact.target.source_blocks[0].uuid
            for candidate in result.triplets:
                subject = (
                    SourceEntity(
                        source_uuid=state.source_uuid,
                        source_block_uuid=source_block_uuid,
                        name=candidate.subject,
                    )
                    if candidate.subject_kind
                    is SourceTripletEndpointKind.ENTITY
                    else SourceEvent(
                        source_uuid=state.source_uuid,
                        source_block_uuid=source_block_uuid,
                        name=candidate.subject,
                    )
                )
                object_ = (
                    SourceEntity(
                        source_uuid=state.source_uuid,
                        source_block_uuid=source_block_uuid,
                        name=candidate.object,
                    )
                    if candidate.object_kind is SourceTripletEndpointKind.ENTITY
                    else SourceEvent(
                        source_uuid=state.source_uuid,
                        source_block_uuid=source_block_uuid,
                        name=candidate.object,
                    )
                )
                predicate = SourcePredicate(
                    source_uuid=state.source_uuid,
                    source_block_uuid=source_block_uuid,
                    predicate=candidate.predicate,
                )
                triplet_occurrences.append(
                    SourceTripletOccurrence(
                        fact=fact,
                        triplet=SourceTriplet(),
                        subject=subject,
                        object=object_,
                        predicate=predicate,
                    )
                )
        return {'triplet_occurrences': triplet_occurrences}
