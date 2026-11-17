"""Bound demand-driven reasoning to complete supplied knowledge context."""

from functools import partial

from kms.concurrency import run_blocking_work
from kms.core.tokenization.budget import TokenBudget
from kms.inference.generation import (
    fits_generation_request,
    require_generation_capacity,
)
from kms.knowledge.models import Argument, KnowledgeContext, Proposition
from kms.knowledge.reasoning.inference import KnowledgeReasoningModule
from kms.knowledge.reasoning.models import ReasoningCandidates, ReasoningRequest
from kms.knowledge.state import KnowledgeState


class KnowledgeReasoningNode:
    """Reason over ranked matching claims and bounded complete arguments."""

    def __init__(
        self,
        module: KnowledgeReasoningModule,
        budget: TokenBudget,
        *,
        candidate_limit: int,
        argument_limit: int,
    ) -> None:
        self._module = module
        self._budget = budget
        self._candidate_limit = candidate_limit
        self._argument_limit = argument_limit

    async def run(
        self, state: KnowledgeState
    ) -> dict[str, ReasoningCandidates | bool]:
        understanding = state['understanding']
        context = state['context']
        if not context.matched:
            return {
                'pending': ReasoningCandidates(
                    new_propositions=[], new_arguments=[]
                ),
                'has_reasoned': True,
            }

        ranked = _ranked_propositions(context)
        by_uuid = {proposition.uuid: proposition for proposition in ranked}
        selected: list[Proposition] = []
        selected_arguments: list[Argument] = []
        for proposition in context.matched:
            if len(selected) == self._candidate_limit:
                break
            candidate = [*selected, proposition]
            request = _request(understanding, context, candidate, [])
            if await self._fits(request):
                selected = candidate
            elif not selected:
                await run_blocking_work(
                    partial(
                        require_generation_capacity,
                        {'request': request},
                        budget=self._budget,
                        image_budget=None,
                    )
                )

        selected_uuids = {proposition.uuid for proposition in selected}
        for argument in context.arguments:
            if len(selected_arguments) == self._argument_limit:
                break
            references = [*argument.premise_uuids, argument.conclusion_uuid]
            missing = list(
                dict.fromkeys(
                    uuid for uuid in references if uuid not in selected_uuids
                )
            )
            if len(selected) + len(missing) > self._candidate_limit:
                continue
            candidate = [*selected, *(by_uuid[uuid] for uuid in missing)]
            arguments = [*selected_arguments, argument]
            request = _request(understanding, context, candidate, arguments)
            if await self._fits(request):
                selected = candidate
                selected_uuids.update(missing)
                selected_arguments = arguments

        for proposition in ranked:
            if len(selected) == self._candidate_limit:
                break
            if proposition.uuid in selected_uuids:
                continue
            candidate = [*selected, proposition]
            request = _request(
                understanding, context, candidate, selected_arguments
            )
            if await self._fits(request):
                selected = candidate
                selected_uuids.add(proposition.uuid)

        request = _request(understanding, context, selected, selected_arguments)
        return {
            'pending': await self._module.acall(request=request),
            'has_reasoned': True,
        }

    async def _fits(self, request: ReasoningRequest) -> bool:
        return await run_blocking_work(
            partial(
                fits_generation_request,
                {'request': request},
                budget=self._budget,
                image_budget=None,
            )
        )


def _ranked_propositions(context: KnowledgeContext) -> list[Proposition]:
    """Put matched propositions first while retaining context order thereafter."""
    ranked = list(context.matched)
    included = {proposition.uuid for proposition in ranked}
    ranked.extend(
        proposition
        for proposition in context.propositions
        if proposition.uuid not in included
    )
    return ranked


def _request(
    understanding: str,
    context: KnowledgeContext,
    propositions: list[Proposition],
    arguments: list[Argument],
) -> ReasoningRequest:
    included = {proposition.uuid for proposition in propositions}
    return ReasoningRequest(
        understanding=understanding,
        context=KnowledgeContext(
            matched=[
                proposition
                for proposition in context.matched
                if proposition.uuid in included
            ],
            propositions=propositions,
            arguments=arguments,
        ),
    )
