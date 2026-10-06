"""Generate descriptions for typed predicate occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_semantic.source_predicate import (
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionResult,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_predicate_description import (
    SourcePredicateDescriptionModule,
)


class SourcePredicateDescriptionWorkerState(TypedDict):
    """State supplied to one predicate description worker."""

    source_predicate_description_request: SourcePredicateDescriptionRequest


class SourcePredicateDescriptionNode:
    """Dispatch and collect one description per predicate occurrence."""

    def __init__(self, module: SourcePredicateDescriptionModule) -> None:
        self._module = module

    def dispatch(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_predicate_description_collect']:
        """Dispatch predicate requests in deterministic source order."""
        if not state.source_predicate_description_requests:
            return 'source_predicate_description_collect'
        return [
            Send(
                'source_predicate_description_worker',
                {'source_predicate_description_request': request},
            )
            for request in state.source_predicate_description_requests
        ]

    async def worker(
        self, state: SourcePredicateDescriptionWorkerState
    ) -> dict[str, list[SourcePredicateDescriptionResult]]:
        """Generate one description for one predicate occurrence."""
        request = state['source_predicate_description_request']
        description = await self._module.acall(request=request.model_input)
        return {
            'source_predicate_description_results': [
                SourcePredicateDescriptionResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SourceSemanticState) -> dict[str, object]:
        """Finish description fan-in before embedding dispatch."""
        return {}
