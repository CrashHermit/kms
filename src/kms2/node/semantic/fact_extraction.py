"""LangGraph workers for source-faithful fact extraction."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.inference import ContextWindowSettings
from kms2.core.model import (
    ExtractedFact,
    FactExtractionRequest,
    FactExtractionResult,
)
from kms2.core.windowing import select_window
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.fact_extraction import FactExtractorModule


class FactWorkerState(TypedDict):
    """State supplied to one fact-extraction worker."""

    fact_extraction_request: FactExtractionRequest


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
                                backward_budget=self._context_window.backward_budget,
                                forward_budget=self._context_window.forward_budget,
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


__all__ = ['FactExtractionNode', 'FactWorkerState']
