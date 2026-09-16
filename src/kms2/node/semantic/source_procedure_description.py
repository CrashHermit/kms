"""Generate descriptions for typed source procedures in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import (
    SourceProcedureDescriptionRequest,
    SourceProcedureDescriptionResult,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_procedure_description import (
    SourceProcedureDescriptionModule,
)


class SourceProcedureDescriptionWorkerState(TypedDict):
    """State supplied to one procedure description worker."""

    source_procedure_description_request: SourceProcedureDescriptionRequest


class SourceProcedureDescriptionNode:
    """Dispatch and collect one description per source procedure."""

    def __init__(self, module: SourceProcedureDescriptionModule) -> None:
        self._module = module

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_procedure_description_collect']:
        """Dispatch procedure requests in deterministic source order."""
        if not state.source_procedure_description_requests:
            return 'source_procedure_description_collect'
        return [
            Send(
                'source_procedure_description_worker',
                {'source_procedure_description_request': request},
            )
            for request in state.source_procedure_description_requests
        ]

    async def worker(
        self, state: SourceProcedureDescriptionWorkerState
    ) -> dict[str, list[SourceProcedureDescriptionResult]]:
        """Generate one description for one source procedure."""
        request = state['source_procedure_description_request']
        description = await self._module.aforward(request=request.model_input)
        return {
            'source_procedure_description_results': [
                SourceProcedureDescriptionResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish procedure description fan-in before embedding dispatch."""
        return {}
