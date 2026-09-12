"""Enrich typed predicate occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import (
    PredicateEnrichmentRequest,
    PredicateEnrichmentResult,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.predicate_enrichment import PredicateEnrichmentModule


class PredicateEnrichmentWorkerState(TypedDict):
    """State supplied to one predicate enrichment worker."""

    predicate_enrichment_request: PredicateEnrichmentRequest


class PredicateEnrichmentNode:
    """Dispatch and collect one description per predicate occurrence."""

    def __init__(self, module: PredicateEnrichmentModule) -> None:
        self._module = module

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['predicate_enrichment_collect']:
        """Dispatch predicate requests in deterministic source order."""
        if not state.predicate_requests:
            return 'predicate_enrichment_collect'
        return [
            Send(
                'predicate_enrichment_worker',
                {'predicate_enrichment_request': request},
            )
            for request in state.predicate_requests
        ]

    async def worker(
        self, state: PredicateEnrichmentWorkerState
    ) -> dict[str, list[PredicateEnrichmentResult]]:
        """Generate one description for one predicate occurrence."""
        request = state['predicate_enrichment_request']
        description = await self._module.aforward(
            request=request.model_input,
        )
        return {
            'predicate_description_results': [
                PredicateEnrichmentResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish description fan-in before embedding dispatch."""
        return {}
