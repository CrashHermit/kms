"""Generate descriptions for typed source statements in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import (
    SourceStatementDescriptionRequest,
    SourceStatementDescriptionResult,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_statement_description import (
    SourceStatementDescriptionModule,
)


class SourceStatementDescriptionWorkerState(TypedDict):
    """State supplied to one statement description worker."""

    source_statement_description_request: SourceStatementDescriptionRequest


class SourceStatementDescriptionNode:
    """Dispatch and collect one description per source statement."""

    def __init__(self, module: SourceStatementDescriptionModule) -> None:
        self._module = module

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_statement_description_collect']:
        """Dispatch statement requests in deterministic source order."""
        if not state.source_statement_description_requests:
            return 'source_statement_description_collect'
        return [
            Send(
                'source_statement_description_worker',
                {'source_statement_description_request': request},
            )
            for request in state.source_statement_description_requests
        ]

    async def worker(
        self, state: SourceStatementDescriptionWorkerState
    ) -> dict[str, list[SourceStatementDescriptionResult]]:
        """Generate one description for one source statement."""
        request = state['source_statement_description_request']
        description = await self._module.aforward(request=request.model_input)
        return {
            'source_statement_description_results': [
                SourceStatementDescriptionResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish statement description fan-in before embedding dispatch."""
        return {}
