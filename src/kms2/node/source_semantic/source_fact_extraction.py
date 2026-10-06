"""LangGraph workers for source-faithful fact extraction."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.source_semantic.source_fact_extraction import (
    SourceFact,
    SourceFactContext,
    SourceFactExtractionRequest,
    SourceFactExtractionResult,
    SourceFactTarget,
)
from kms2.core.windowing import (
    TextTokenCounter,
    count_text_tokens,
    select_window,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_fact_extraction import (
    SourceFactExtractorModule,
)


class SourceFactWorkerState(TypedDict):
    """State supplied to one fact-extraction worker."""

    fact_extraction_request: SourceFactExtractionRequest


class SourceFactExtractionNode:
    """Dispatch, run, and collect atomic fact extraction workers."""

    def __init__(
        self,
        extractor: SourceFactExtractorModule,
        context_window: ContextWindowSettings,
        *,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._extractor = extractor
        self._context_window = context_window
        self._token_counters = token_counters

    async def dispatch(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_fact_collect']:
        """Dispatch one bounded target selection per canonical source block."""
        token_counts = count_text_tokens(
            [block.content for block in state.blocks], self._token_counters
        )
        sends: list[Send] = []
        for position, _ in enumerate(state.blocks):
            window = select_window(
                state.blocks,
                [position],
                token_counts=token_counts,
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=self._context_window.target_budget,
            )
            target = SourceFactTarget(
                source_blocks=[
                    state.blocks[position + offset]
                    for offset in range(len(window.target))
                ]
            )
            context_before = SourceFactContext(
                source_blocks=[
                    state.blocks[index]
                    for index in range(
                        position - len(window.context_before), position
                    )
                ]
            )
            target_end = position + len(target.source_blocks)
            context_after = SourceFactContext(
                source_blocks=state.blocks[
                    target_end : target_end + len(window.context_after)
                ]
            )
            sends.append(
                Send(
                    'source_fact_worker',
                    {
                        'fact_extraction_request': SourceFactExtractionRequest(
                            source_uuid=state.source_uuid,
                            target=target,
                            context_before=context_before,
                            context_after=context_after,
                        )
                    },
                )
            )
        return sends or 'source_fact_collect'

    async def worker(
        self, state: SourceFactWorkerState
    ) -> dict[str, list[SourceFactExtractionResult]]:
        """Extract facts from one target selection."""
        request = state['fact_extraction_request']
        facts = await self._extractor.aforward(request=request.model_input())
        return {
            'fact_results': [
                SourceFactExtractionResult(
                    target=request.target,
                    context_before=request.context_before,
                    context_after=request.context_after,
                    facts=facts,
                )
            ]
        }

    def collect(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourceFact]]:
        """Collect facts in canonical target order with private evidence nodes."""
        position = {
            block.uuid: index for index, block in enumerate(state.blocks)
        }
        results = sorted(
            state.fact_results,
            key=lambda item: position[item.target.source_blocks[0].uuid],
        )
        source_facts: list[SourceFact] = []
        for result in results:
            for fact in result.facts:
                source_facts.append(
                    SourceFact(
                        text=fact.text,
                        target=SourceFactTarget(
                            source_blocks=list(result.target.source_blocks)
                        ),
                        context_before=SourceFactContext(
                            source_blocks=list(
                                result.context_before.source_blocks
                            )
                        ),
                        context_after=SourceFactContext(
                            source_blocks=list(
                                result.context_after.source_blocks
                            )
                        ),
                    )
                )
        return {'source_facts': source_facts}
