"""Extract source-grounded candidates from an owned unit window."""

from kms.core.tokenization.budget import TokenBudget
from kms.knowledge.extraction.inference import UnitExtractionModule
from kms.knowledge.extraction.models import (
    UnitExtractionRequest,
    UnitExtractionResult,
)
from kms.knowledge.state import KnowledgeState
from kms.media.budget import ImagePayloadBudget
from kms.source.retrieval.models import SourceUnitWindow
from kms.source.retrieval.read import SourceReader


class UnitKnowledgeExtractionNode:
    """Run source-grounded extraction without promoting candidates to evidence."""

    def __init__(
        self,
        reader: SourceReader,
        module: UnitExtractionModule,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
        *,
        before_tokens: int,
        after_tokens: int,
    ) -> None:
        self._reader = reader
        self._module = module
        self._budget = budget
        self._image_budget = image_budget
        self._before_tokens = before_tokens
        self._after_tokens = after_tokens

    async def run(
        self, state: KnowledgeState
    ) -> dict[str, UnitExtractionResult | None]:
        context = state['context']

        def request(window: SourceUnitWindow) -> UnitExtractionRequest:
            return UnitExtractionRequest(
                understanding=state['understanding'],
                context_before=window.before,
                target=window.target,
                context_after=window.after,
                existing_propositions=context.propositions,
                existing_arguments=context.arguments,
            )

        window = await self._reader.read_unit_window(
            state['user_uuid'],
            state['unit_uuid'],
            before_tokens=self._before_tokens,
            after_tokens=self._after_tokens,
            budget=self._budget,
            image_budget=self._image_budget,
            request_inputs=lambda window: {'request': request(window)},
        )
        if window is None:
            return {'pending': None}
        candidates = await self._module.acall(request=request(window))
        return {
            'pending': UnitExtractionResult(
                source_uuid=window.source.uuid,
                target_unit_uuid=window.target.uuid,
                candidates=candidates,
            )
        }
