"""Generate descriptions for source entity occurrences in a LangGraph phase."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Send

from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionItem,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
)
from kms.core.tokenization.budget import TokenBudget
from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.module.source_semantic.source_entity_description import (
    SourceEntityDescriptionModule,
)
from kms.service.content import load_content_images, pack_content_batches


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
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._module = module
        self._budget = budget
        self._batch_size = batch_size
        self._image_budget = image_budget

    async def dispatch(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_description_collect']:
        """Dispatch complete entity requests in bounded source-order packets."""
        requests = state.source_entity_description_requests
        if not requests:
            return 'source_entity_description_collect'
        incoming = [
            requests[index : index + self._batch_size]
            for index in range(0, len(requests), self._batch_size)
        ]
        batches = await pack_content_batches(
            incoming,
            project=_project_items,
            budget=self._budget,
            image_budget=self._image_budget,
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
        builder = MultimodalContentBuilder()
        items = _project_items(requests, builder)['items']
        payloads = await load_content_images(
            builder.images, budget=self._image_budget
        )
        outputs = await self._module.acall(
            items=items,
            content_images=[dspy.Image(payload) for payload in payloads],
        )
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


def _project_items(
    requests: list[SourceEntityDescriptionRequest],
    builder: MultimodalContentBuilder,
) -> dict[str, object]:
    """Project one batch in its own ordered image catalog."""
    return {
        'items': [
            SourceEntityDescriptionItem(
                position=position,
                request=request.model_input(builder),
            )
            for position, request in enumerate(requests)
        ]
    }
