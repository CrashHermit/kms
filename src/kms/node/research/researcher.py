"""Build a bounded text context and run one researcher prediction."""

from kms.core.model.research.inference import (
    FinishResearchAction,
    ResearchContext,
    ResearchResult,
)
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.langgraph.teaching.state import TeachingTurnState
from kms.module.research.researcher import ResearcherModule


class ResearcherNode:
    """Present accumulated search results under the configured token budget."""

    def __init__(
        self, researcher: ResearcherModule, token_budget: TokenBudget
    ) -> None:
        self._researcher = researcher
        self._token_budget = token_budget

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        assert state.research_task is not None
        if state.research_tool_calls >= state.research_max_tool_calls:
            state.latest_action = FinishResearchAction(kind='finish')
            state.final_research_result = self._final_result(state)
            return {
                'latest_action': state.latest_action,
                'final_research_result': state.final_research_result,
            }
        context = ResearchContext(
            task=state.research_task,
            search_results=state.search_results.copy(),
            observations=state.research_observations.copy(),
            tools_remaining=max(
                0, state.research_max_tool_calls - state.research_tool_calls
            ),
        )
        if (
            self._token_budget.counter.estimate(text=context.model_dump_json())
            > self._token_budget.token_limit
        ):
            raise InputBudgetExceeded(
                'Complete research context exceeds the configured input token budget'
            )
        state.latest_action = await self._researcher.aforward(context=context)
        if isinstance(state.latest_action, FinishResearchAction):
            state.final_research_result = self._final_result(state)
        return {
            'latest_action': state.latest_action,
            'final_research_result': state.final_research_result,
        }

    @staticmethod
    def _final_result(state: TeachingTurnState) -> ResearchResult:
        return ResearchResult(
            source_revisions=state.persisted_source_revisions.copy(),
            failures=state.failures.copy(),
        )


__all__ = ['ResearcherNode']
