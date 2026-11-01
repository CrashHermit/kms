"""Run grounded proposition extraction for one completed teaching response."""

import dspy

from kms2.core.model.knowledge.admission import KnowledgeDraft
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.knowledge.state import KnowledgeState
from kms2.module.knowledge.extraction import KnowledgeExtractorModule


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
        token_count = self._budget.counter.count_texts([payload])[0]
        if token_count > self._budget.token_limit:
            raise InputBudgetExceeded(
                'Knowledge extraction evidence exceeds the configured input budget'
            )
        images = [dspy.Image(data) for data in state.evidence_image_bytes]
        draft = await self._module.acall(
            request=state.request,
            evidence_images=images,
        )
        return {'draft': draft}
