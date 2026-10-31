"""Discover and conservatively group source-local proposition hubs."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.source_semantic import SourcePropositionHubSettings
from kms2.core.model.source_semantic.source_proposition_hub import (
    SourcePropositionHub,
    SourcePropositionHubCandidate,
    SourcePropositionHubGroup,
    SourcePropositionHubJudgeInput,
    SourcePropositionHubJudgeResult,
    SourcePropositionHubMember,
    SourcePropositionHubRerankResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import TokenBudget, pack_items
from kms2.database.source_semantic.source_proposition_repository import (
    SourcePropositionRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_proposition_hub_judge import (
    SourcePropositionHubJudgeModule,
)


class SourcePropositionHubRerankWorkerState(TypedDict):
    """State supplied to one reranker worker."""

    source_proposition_hub_rerank_ordinal: int
    source_proposition_hub_left_text: str
    source_proposition_hub_batch: list[SourcePropositionHubCandidate]


class SourcePropositionHubJudgeWorkerState(TypedDict):
    """State supplied to one equivalence-judge worker."""

    source_proposition_hub_judge_ordinal: int
    source_proposition_hub_batch: list[SourcePropositionHubCandidate]


class SourcePropositionHubNode:
    """Run candidate reranking, explicit judging, and complete-link grouping."""

    def __init__(
        self,
        repository: SourcePropositionRepository,
        judge_module: SourcePropositionHubJudgeModule,
        reranker: RerankerClient,
        settings: SourcePropositionHubSettings,
        *,
        reranker_budget: TokenBudget,
        judge_budget: TokenBudget,
    ) -> None:
        self._repository = repository
        self._judge_module = judge_module
        self._reranker = reranker
        self._settings = settings
        self._reranker_budget = reranker_budget
        self._judge_budget = judge_budget

    async def load_candidates(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Load every source occurrence and its bounded vector candidates."""
        candidates = (
            await self._repository.read_source_proposition_hub_candidates(
                state.source_uuid,
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        members = await self._repository.read_source_proposition_hub_members(
            state.source_uuid
        )
        return {
            'source_proposition_hub_candidates': candidates,
            'source_proposition_hub_members': members,
        }

    def dispatch_rerank(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_proposition_hub_rerank_collect']:
        """Pack source candidate pairs into bounded reranker requests."""
        if not state.source_proposition_hub_candidates:
            return 'source_proposition_hub_rerank_collect'
        grouped: dict[str, list[SourcePropositionHubCandidate]] = {}
        for candidate in state.source_proposition_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        sends: list[Send] = []
        ordinal = 0
        for candidates in grouped.values():
            left_text = candidates[0].left_text
            pair_texts = [
                _pair_text(left_text, candidate.right_text)
                for candidate in candidates
            ]
            costs = self._reranker_budget.counter.count_texts(pair_texts)
            batches = pack_items(
                candidates,
                token_counts=costs,
                token_budget=self._reranker_budget.token_limit,
            )
            for batch in batches:
                sends.append(
                    Send(
                        'source_proposition_hub_rerank_worker',
                        {
                            'source_proposition_hub_rerank_ordinal': ordinal,
                            'source_proposition_hub_left_text': left_text,
                            'source_proposition_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: SourcePropositionHubRerankWorkerState
    ) -> dict[str, list[SourcePropositionHubRerankResult]]:
        """Retain every candidate above the reranker rejection threshold."""
        batch = state['source_proposition_hub_batch']
        results = await self._reranker.rerank(
            state['source_proposition_hub_left_text'],
            [candidate.right_text for candidate in batch],
            top_n=None,
        )
        retained = [
            batch[result['index']]
            for result in results
            if result['relevance_score']
            >= self._settings.reranker_rejection_threshold
        ]
        return {
            'source_proposition_hub_rerank_results': [
                SourcePropositionHubRerankResult(
                    ordinal=state['source_proposition_hub_rerank_ordinal'],
                    candidates=retained,
                )
            ]
        }

    def collect_rerank(self, state: SourceSemanticState) -> dict[str, object]:
        """Collect reranker survivors in stable worker order."""
        candidates = [
            candidate
            for result in sorted(
                state.source_proposition_hub_rerank_results,
                key=lambda item: item.ordinal,
            )
            for candidate in result.candidates
        ]
        return {'source_proposition_hub_reranked_candidates': candidates}

    def dispatch_judge(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_proposition_hub_judge_collect']:
        """Batch every reranker-surviving pair for explicit judgment."""
        candidates = state.source_proposition_hub_reranked_candidates
        if not candidates:
            return 'source_proposition_hub_judge_collect'
        costs = self._judge_budget.counter.count_texts(
            [
                _judge_input(index, candidate).model_dump_json()
                for index, candidate in enumerate(candidates)
            ]
        )
        batches = pack_items(
            candidates,
            token_counts=costs,
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'source_proposition_hub_judge_worker',
                {
                    'source_proposition_hub_judge_ordinal': ordinal,
                    'source_proposition_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: SourcePropositionHubJudgeWorkerState
    ) -> dict[str, list[SourcePropositionHubJudgeResult]]:
        """Judge every pair in one batch and retain explicit positives."""
        batch = state['source_proposition_hub_batch']
        decisions = await self._judge_module.acall(
            requests=[
                _judge_input(index, item) for index, item in enumerate(batch)
            ]
        )
        accepted = [
            batch[decision.index]
            for decision in decisions
            if decision.equivalent
        ]
        return {
            'source_proposition_hub_judge_results': [
                SourcePropositionHubJudgeResult(
                    ordinal=state['source_proposition_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_groups(self, state: SourceSemanticState) -> dict[str, object]:
        """Apply explicit-positive complete-link merges and retain singletons."""
        accepted = [
            candidate
            for result in sorted(
                state.source_proposition_hub_judge_results,
                key=lambda item: item.ordinal,
            )
            for candidate in result.accepted
        ]
        groups = _complete_link_groups(
            state.source_proposition_hub_members, accepted
        )
        return {'source_proposition_hub_groups': groups}

    def prepare_hubs(self, state: SourceSemanticState) -> dict[str, object]:
        """Create identity-only hubs without copying or embedding member text."""
        groups = state.source_proposition_hub_groups
        return {
            'source_proposition_hubs': [SourcePropositionHub() for _ in groups],
            'source_proposition_hub_memberships': [
                [member.uuid for member in group.members] for group in groups
            ],
        }


def _complete_link_groups(
    members: list[SourcePropositionHubMember],
    accepted: list[SourcePropositionHubCandidate],
) -> list[SourcePropositionHubGroup]:
    """Merge only groups whose every cross-pair was explicitly approved."""
    member_rank = {member.uuid: rank for rank, member in enumerate(members)}
    groups: list[set[str]] = [{member.uuid} for member in members]
    member_by_uuid = {member.uuid: member for member in members}
    positive_pairs = {
        (candidate.left_uuid, candidate.right_uuid) for candidate in accepted
    }
    for candidate in sorted(
        accepted,
        key=lambda item: (-item.score, item.left_uuid, item.right_uuid),
    ):
        left_group = next(
            group for group in groups if candidate.left_uuid in group
        )
        right_group = next(
            group for group in groups if candidate.right_uuid in group
        )
        if left_group is right_group:
            continue
        if not all(
            (min(left_uuid, right_uuid), max(left_uuid, right_uuid))
            in positive_pairs
            for left_uuid in left_group
            for right_uuid in right_group
        ):
            continue
        left_group.update(right_group)
        groups.remove(right_group)
    final_groups = [
        sorted(
            (member_by_uuid[uuid] for uuid in group),
            key=lambda member: member_rank[member.uuid],
        )
        for group in groups
    ]
    final_groups.sort(key=lambda group: member_rank[group[0].uuid])
    return [
        SourcePropositionHubGroup(ordinal=index, members=group)
        for index, group in enumerate(final_groups)
    ]


def _pair_text(left_text: str, right_text: str) -> str:
    """Render a candidate pair as one budgeted input."""
    return f'Left: {left_text}\nRight: {right_text}'


def _judge_input(
    index: int, candidate: SourcePropositionHubCandidate
) -> SourcePropositionHubJudgeInput:
    """Build one zero-based proposition equivalence request."""
    return SourcePropositionHubJudgeInput(
        index=index,
        left_text=candidate.left_text,
        right_text=candidate.right_text,
    )
