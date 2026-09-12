"""LangGraph workers for two-pass source-faithful triplet extraction."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config import ContextWindowSettings
from kms2.core.context_window import select_window
from kms2.core.model import (
    Entity,
    Event,
    ExtractedFact,
    FactExtractionRequest,
    FactExtractionResult,
    Predicate,
    RawAssertion,
    RawTriplet,
    SemanticNodeKind,
    TripletDecompositionRequest,
    TripletDecompositionResult,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.triplet import (
    FactExtractorModule,
    TripletDecomposerModule,
)


class FactWorkerState(TypedDict):
    """State supplied to one fact-extraction worker."""

    fact_extraction_request: FactExtractionRequest


class TripletWorkerState(TypedDict):
    """State supplied to one triplet-decomposition worker."""

    triplet_decomposition_request: TripletDecompositionRequest


class FactExtractionNode:
    """Dispatch, run, and collect atomic fact extraction workers."""

    def __init__(
        self,
        extractor: FactExtractorModule,
        context_window: ContextWindowSettings,
    ) -> None:
        self._extractor = extractor
        self._context_window = context_window

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['fact_collect']:
        """Dispatch one authoritative target block per persisted source block."""
        sends: list[Send] = []
        blocks = state.blocks
        for position, target in enumerate(blocks):
            sends.append(
                Send(
                    'fact_worker',
                    {
                        'fact_extraction_request': FactExtractionRequest(
                            source_uuid=state.source_uuid,
                            target_block=target,
                            window=select_window(
                                blocks,
                                [position],
                                backward_budget=(
                                    self._context_window.backward_budget
                                ),
                                forward_budget=(
                                    self._context_window.forward_budget
                                ),
                            ),
                        )
                    },
                )
            )
        return sends or 'fact_collect'

    async def worker(
        self, state: FactWorkerState
    ) -> dict[str, list[FactExtractionResult]]:
        """Extract facts from one target block."""
        request = state['fact_extraction_request']
        facts = await self._extractor.aforward(request=request.model_input())
        return {
            'fact_results': [
                FactExtractionResult(
                    target_block_uuid=request.target_block.uuid,
                    facts=[
                        ExtractedFact(
                            source_uuid=request.source_uuid,
                            source_block_uuid=request.target_block.uuid,
                            text=fact.text,
                        )
                        for fact in facts
                    ],
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, list[ExtractedFact]]:
        """Collect facts in persisted source-block order."""
        position = {
            block.uuid: index for index, block in enumerate(state.blocks)
        }
        facts: list[ExtractedFact] = []
        for result in sorted(
            state.fact_results,
            key=lambda item: position[item.target_block_uuid],
        ):
            facts.extend(result.facts)
        return {'extracted_facts': facts}


class TripletDecompositionNode:
    """Dispatch facts, collect candidates, and create raw occurrences."""

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

    def collect(self, state: SemanticState) -> dict[str, list[RawAssertion]]:
        """Materialize each candidate as a fresh raw assertion occurrence."""
        results = sorted(
            state.triplet_results,
            key=lambda item: item.fact_index,
        )
        assertions: list[RawAssertion] = []
        for result in results:
            fact = result.fact
            for candidate in result.triplets:
                subject = (
                    Entity(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.subject,
                    )
                    if candidate.subject_kind is SemanticNodeKind.ENTITY
                    else Event(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.subject,
                    )
                )
                object_ = (
                    Entity(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.object,
                    )
                    if candidate.object_kind is SemanticNodeKind.ENTITY
                    else Event(
                        source_uuid=fact.source_uuid,
                        source_block_uuid=fact.source_block_uuid,
                        name=candidate.object,
                    )
                )
                predicate = Predicate(
                    source_uuid=fact.source_uuid,
                    source_block_uuid=fact.source_block_uuid,
                    predicate=candidate.predicate,
                )
                triplet = RawTriplet(
                    source_uuid=fact.source_uuid,
                    source_block_uuid=fact.source_block_uuid,
                    subject_uuid=subject.uuid,
                    object_uuid=object_.uuid,
                    predicate_uuid=predicate.uuid,
                )
                assertions.append(
                    RawAssertion(
                        triplet=triplet,
                        subject=subject,
                        object=object_,
                        predicate=predicate,
                    )
                )
        return {'raw_assertions': assertions}
