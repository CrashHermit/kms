"""Enrich typed entity occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import EntityEnrichmentRequest, EntityEnrichmentResult
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.entity_enrichment import EntityEnrichmentModule


class EntityEnrichmentWorkerState(TypedDict):
    """State supplied to one entity enrichment worker."""

    entity_enrichment_request: EntityEnrichmentRequest


class EntityEnrichmentNode:
    """Dispatch and collect one description per entity occurrence."""

    def __init__(self, module: EntityEnrichmentModule) -> None:
        self._module = module

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['entity_enrichment_collect']:
        """Dispatch entity requests in deterministic source order."""
        if not state.entity_requests:
            return 'entity_enrichment_collect'
        return [
            Send(
                'entity_enrichment_worker',
                {'entity_enrichment_request': request},
            )
            for request in state.entity_requests
        ]

    async def worker(
        self, state: EntityEnrichmentWorkerState
    ) -> dict[str, list[EntityEnrichmentResult]]:
        """Generate one description for one entity occurrence."""
        request = state['entity_enrichment_request']
        description = await self._module.aforward(
            request=request.model_input,
        )
        return {
            'entity_description_results': [
                EntityEnrichmentResult(
                    **request.target.model_dump(),
                    description=description,
                )
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish description fan-in before embedding dispatch."""
        return {}
