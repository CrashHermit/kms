"""Generate descriptions for typed event occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import (
    SourceEventDescriptionRequest,
    SourceEventDescriptionResult,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_event_description import (
    SourceEventDescriptionModule,
)


class SourceEventDescriptionWorkerState(TypedDict):
    """State supplied to one event description worker."""

    source_event_description_request: SourceEventDescriptionRequest


class SourceEventDescriptionNode:
    """Dispatch and collect one description per event occurrence."""

    def __init__(self, module: SourceEventDescriptionModule) -> None:
        self._module = module

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_event_description_collect']:
        """Dispatch event requests in deterministic source order."""
        if not state.source_event_description_requests:
            return 'source_event_description_collect'
        return [
            Send(
                'source_event_description_worker',
                {'source_event_description_request': request},
            )
            for request in state.source_event_description_requests
        ]

    async def worker(
        self, state: SourceEventDescriptionWorkerState
    ) -> dict[str, list[SourceEventDescriptionResult]]:
        """Generate one description for one event occurrence."""
        request = state['source_event_description_request']
        description = await self._module.aforward(
            request=request.model_input,
        )
        return {
            'source_event_description_results': [
                SourceEventDescriptionResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish description fan-in before embedding dispatch."""
        return {}
