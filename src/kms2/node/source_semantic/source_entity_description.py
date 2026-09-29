"""Generate descriptions for typed entity occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_entity_description import (
    SourceEntityDescriptionModule,
)


class SourceEntityDescriptionWorkerState(TypedDict):
    """State supplied to one entity description worker."""

    source_entity_description_request: SourceEntityDescriptionRequest


class SourceEntityDescriptionNode:
    """Dispatch and collect one description per entity occurrence."""

    def __init__(self, module: SourceEntityDescriptionModule) -> None:
        self._module = module

    def dispatch(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_description_collect']:
        """Dispatch entity requests in deterministic source order."""
        if not state.source_entity_description_requests:
            return 'source_entity_description_collect'
        return [
            Send(
                'source_entity_description_worker',
                {'source_entity_description_request': request},
            )
            for request in state.source_entity_description_requests
        ]

    async def worker(
        self, state: SourceEntityDescriptionWorkerState
    ) -> dict[str, list[SourceEntityDescriptionResult]]:
        """Generate one description for one entity occurrence."""
        request = state['source_entity_description_request']
        description = await self._module.aforward(
            request=request.model_input,
        )
        return {
            'source_entity_description_results': [
                SourceEntityDescriptionResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SourceSemanticState) -> dict[str, object]:
        """Finish description fan-in before embedding dispatch."""
        return {}
