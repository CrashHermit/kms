"""LangGraph workers for independent proposition extraction."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.source_extraction import (
    SourceExtractionRequest,
    SourceExtractionTarget,
)
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourceProposition,
    SourcePropositionExtractionResult,
)
from kms2.core.windowing import TextTokenCounter
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_proposition_extraction import (
    SourcePropositionExtractorModule,
)
from kms2.node.source_extraction import (
    build_source_extraction_requests,
)


class SourcePropositionWorkerState(TypedDict):
    """State supplied to one proposition extraction worker."""

    proposition_extraction_request: SourceExtractionRequest


class SourcePropositionExtractionNode:
    """Dispatch, execute, and collect proposition extraction workers."""

    def __init__(
        self,
        extractor: SourcePropositionExtractorModule,
        context_window: ContextWindowSettings,
        *,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._extractor = extractor
        self._context_window = context_window
        self._token_counters = token_counters

    def dispatch_propositions(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_proposition_collect']:
        """Dispatch independently bounded target requests."""
        requests = build_source_extraction_requests(
            state.blocks,
            source_uuid=state.source_uuid,
            context_window=self._context_window,
            token_counters=self._token_counters,
        )
        if not requests:
            return 'source_proposition_collect'
        return [
            Send(
                'source_proposition_worker',
                {'proposition_extraction_request': request},
            )
            for request in requests
        ]

    async def worker(
        self, state: SourcePropositionWorkerState
    ) -> dict[str, list[SourcePropositionExtractionResult]]:
        """Extract claims from one request's target evidence."""
        request = state['proposition_extraction_request']
        propositions = await self._extractor.acall(
            request=request.model_input()
        )
        return {
            'proposition_results': [
                SourcePropositionExtractionResult(
                    request=request, propositions=propositions
                )
            ]
        }

    def collect(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourceProposition]]:
        """Collect source-ordered occurrence results."""
        positions = {
            block.uuid: index for index, block in enumerate(state.blocks)
        }
        results = sorted(
            state.proposition_results,
            key=lambda result: positions[
                result.request.target.source_blocks[0].uuid
            ],
        )
        propositions: list[SourceProposition] = []
        for result in results:
            request = result.request
            for candidate in sorted(
                result.propositions,
                key=lambda candidate: candidate.source_position,
            ):
                owner_position = candidate.source_position
                owner = request.target.source_blocks[owner_position]
                propositions.append(
                    SourceProposition(
                        source_uuid=request.source_uuid,
                        text=candidate.text,
                        target=SourceExtractionTarget(source_blocks=[owner]),
                    )
                )
        return {'source_propositions': propositions}
