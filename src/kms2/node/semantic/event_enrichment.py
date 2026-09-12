"""Enrich typed event occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import EventEnrichmentRequest, EventEnrichmentResult
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.event_enrichment import EventEnrichmentModule


class EventEnrichmentWorkerState(TypedDict):
    """State supplied to one event enrichment worker."""

    event_enrichment_request: EventEnrichmentRequest


class EventEnrichmentNode:
    """Dispatch and collect one description per event occurrence."""

    def __init__(self, module: EventEnrichmentModule) -> None:
        self._module = module

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['event_enrichment_collect']:
        """Dispatch event requests in deterministic source order."""
        if not state.event_requests:
            return 'event_enrichment_collect'
        return [
            Send(
                'event_enrichment_worker',
                {'event_enrichment_request': request},
            )
            for request in state.event_requests
        ]

    async def worker(
        self, state: EventEnrichmentWorkerState
    ) -> dict[str, list[EventEnrichmentResult]]:
        """Generate one description for one event occurrence."""
        request = state['event_enrichment_request']
        description = await self._module.aforward(
            request=request.model_input,
        )
        return {
            'event_description_results': [
                EventEnrichmentResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish description fan-in before embedding dispatch."""
        return {}
