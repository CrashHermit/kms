"""LangGraph workers for independent entity extraction."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.source_extraction import (
    SourceExtractionRequest,
    SourceExtractionTarget,
)
from kms2.core.model.source_semantic.source_entity import SourceEntity
from kms2.core.model.source_semantic.source_entity_extraction import (
    SourceEntityExtractionResult,
)
from kms2.core.windowing import TextTokenCounter
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_entity_extraction import (
    SourceEntityExtractorModule,
)
from kms2.node.source_extraction import build_source_extraction_requests


class SourceEntityWorkerState(TypedDict):
    """State supplied to one entity extraction worker."""

    entity_extraction_request: SourceExtractionRequest


class SourceEntityExtractionNode:
    """Dispatch, execute, and collect entity extraction workers."""

    def __init__(
        self,
        extractor: SourceEntityExtractorModule,
        context_window: ContextWindowSettings,
        *,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._extractor = extractor
        self._context_window = context_window
        self._token_counters = token_counters

    def dispatch_entities(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_extraction_collect']:
        """Dispatch independently bounded target requests."""
        requests = build_source_extraction_requests(
            state.blocks,
            source_uuid=state.source_uuid,
            context_window=self._context_window,
            token_counters=self._token_counters,
        )
        if not requests:
            return 'source_entity_extraction_collect'
        return [
            Send(
                'source_entity_extraction_worker',
                {'entity_extraction_request': request},
            )
            for request in requests
        ]

    async def worker(
        self, state: SourceEntityWorkerState
    ) -> dict[str, list[SourceEntityExtractionResult]]:
        """Extract entities from one request's target evidence."""
        request = state['entity_extraction_request']
        entities = await self._extractor.acall(request=request.model_input())
        return {
            'source_entity_extraction_results': [
                SourceEntityExtractionResult(request=request, entities=entities)
            ]
        }

    def collect(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourceEntity]]:
        """Collect source-ordered block-owned entity occurrences."""
        positions = {
            block.uuid: index for index, block in enumerate(state.blocks)
        }
        results = sorted(
            state.source_entity_extraction_results,
            key=lambda result: positions[
                result.request.target.source_blocks[0].uuid
            ],
        )
        entities: list[SourceEntity] = []
        for result in results:
            request = result.request
            for candidate in sorted(
                result.entities, key=lambda candidate: candidate.source_position
            ):
                owner = request.target.source_blocks[candidate.source_position]
                entities.append(
                    SourceEntity(
                        source_uuid=request.source_uuid,
                        name=candidate.name,
                        target=SourceExtractionTarget(source_blocks=[owner]),
                    )
                )
        return {'source_entities': entities}
