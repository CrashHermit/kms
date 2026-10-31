"""Generate descriptions for source entity occurrences in a LangGraph phase."""

import json
from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionItem,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
)
from kms2.core.windowing import TokenBudget, pack_items
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_entity_description import (
    SourceEntityDescriptionModule,
)


class SourceEntityDescriptionWorkerState(TypedDict):
    """State supplied to one entity-description packet worker."""

    source_entity_description_requests: list[SourceEntityDescriptionRequest]


class SourceEntityDescriptionNode:
    """Dispatch and collect descriptions for source entity occurrences."""

    def __init__(
        self,
        module: SourceEntityDescriptionModule,
        *,
        budget: TokenBudget,
        batch_size: int,
    ) -> None:
        self._module = module
        self._budget = budget
        self._batch_size = batch_size

    def dispatch(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_description_collect']:
        """Dispatch complete entity requests in bounded source-order packets."""
        requests = state.source_entity_description_requests
        if not requests:
            return 'source_entity_description_collect'

        indexed_items = [
            SourceEntityDescriptionItem(
                position=self._batch_size - 1,
                request=request.model_input,
            )
            for request in requests
        ]
        item_texts = [
            json.dumps(
                item.model_dump(mode='json'),
                ensure_ascii=False,
                separators=(',', ':'),
            )
            for item in indexed_items
        ]
        item_costs = self._budget.counter.count_texts(item_texts)
        list_overhead = self._budget.counter.count_texts(['[,]'])[0]
        envelope_cost = self._budget.counter.count_texts(['{"items":[]}'])[0]
        batches = pack_items(
            requests,
            token_counts=[cost + list_overhead for cost in item_costs],
            token_budget=self._budget.token_limit - envelope_cost,
            max_items=self._batch_size,
        )
        return [
            Send(
                'source_entity_description_worker',
                {'source_entity_description_requests': batch},
            )
            for batch in batches
        ]

    async def worker(
        self, state: SourceEntityDescriptionWorkerState
    ) -> dict[str, list[SourceEntityDescriptionResult]]:
        """Describe each entity occurrence and restore request order."""
        requests = state['source_entity_description_requests']
        items = [
            SourceEntityDescriptionItem(
                position=position,
                request=request.model_input,
            )
            for position, request in enumerate(requests)
        ]
        outputs = await self._module.acall(items=items)
        descriptions = {
            output.position: output.description for output in outputs
        }
        return {
            'source_entity_description_results': [
                SourceEntityDescriptionResult(
                    **request.target.model_dump(),
                    description=descriptions[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(self, state: SourceSemanticState) -> dict[str, object]:
        """Finish entity-description fan-in before embedding dispatch."""
        return {}
