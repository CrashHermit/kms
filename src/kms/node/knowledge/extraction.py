"""Run grounded proposition extraction for one completed teaching response."""

import hashlib

import dspy

from kms.core.content import ContentImage
from kms.core.model.knowledge.admission import KnowledgeDraft
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.langgraph.knowledge.state import KnowledgeState
from kms.module.knowledge.extraction import KnowledgeExtractorModule


class KnowledgeExtractionNode:
    """Execute the extraction predictor over exact inspected teaching evidence."""

    def __init__(
        self,
        module: KnowledgeExtractorModule,
        budget: TokenBudget,
    ) -> None:
        self._module = module
        self._budget = budget

    async def run(self, state: KnowledgeState) -> dict[str, KnowledgeDraft]:
        """Extract and store a request-local draft."""
        payload = state.request.model_dump_json()
        content_images = [
            ContentImage(
                key=('content', hashlib.sha256(data).hexdigest()),
                payload=data,
            )
            for data in state.content_image_bytes
        ]
        token_count = self._budget.counter.estimate(
            text=payload, images=content_images
        )
        if token_count > self._budget.token_limit:
            raise InputBudgetExceeded(
                'Knowledge extraction evidence exceeds the configured input budget'
            )
        images = [dspy.Image(image.payload) for image in content_images]
        draft = await self._module.acall(
            request=state.request,
            content_images=images,
        )
        return {'draft': draft}
