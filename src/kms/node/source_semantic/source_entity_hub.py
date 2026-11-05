"""Dispatch and collect iterative source-local entity hub reconciliation."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Overwrite, Send

from kms.config.source_semantic import SourceEntityHubSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source_extraction import SourceEvidenceInput
from kms.core.model.source_semantic.source_entity_hub import (
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubJudgeInput,
    SourceEntityHubJudgeResult,
    SourceEntityHubMember,
    SourceEntityHubRerankResult,
)
from kms.core.reranking import Reranker
from kms.core.source_context import count_source_tokens
from kms.core.tokenization.budget import TokenBudget
from kms.core.windowing import select_window
from kms.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.module.source_semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
)
from kms.service.content import load_content_images, pack_content_batches


class SourceEntityHubRerankWorkerState(TypedDict):
    """State supplied to one entity reranker worker."""

    source_entity_hub_rerank_ordinal: int
    source_entity_hub_left_text: str
    source_entity_hub_candidates: list[SourceEntityHubCandidate]


class SourceEntityHubJudgeWorkerState(TypedDict):
    """State supplied to one entity judge worker."""

    source_entity_hub_judge_ordinal: int
    source_entity_hub_batch: list[SourceEntityHubCandidate]


class SourceEntityHubInitialState(TypedDict):
    """Required minimal entity state used to start reconciliation."""

    source_entity_hub_members: list[SourceEntityHubMember]
    source_entity_hub_groups: list[list[SourceEntityHubMember]]
    source_entity_hub_decision_cache: dict[str, bool]


class SourceEntityHubNode:
    """Reconcile existing entity occurrences within one source semantic stage."""

    def __init__(
        self,
        repository: SourceEntityRepository,
        judge_module: SourceEntityHubJudgeModule,
        reranker: Reranker,
        settings: SourceEntityHubSettings,
        *,
        judge_budget: TokenBudget,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._repository = repository
        self._judge_module = judge_module
        self._reranker = reranker
        self._settings = settings
        self._judge_budget = judge_budget
        self._image_budget = image_budget

    async def initialize(
        self, state: SourceSemanticState
    ) -> SourceEntityHubInitialState:
        """Load members and attach canonical owning context windows."""
        members = await self._repository.read_source_entity_hub_members(
            state.source_uuid
        )
        if members:
            token_counts = count_source_tokens(
                state.blocks, (self._judge_budget.counter,)
            )
            positions = {
                block.uuid: position
                for position, block in enumerate(state.blocks)
            }
            for member in members:
                window = select_window(
                    state.blocks,
                    [positions[member.source_block_uuid]],
                    token_counts=token_counts,
                    backward_budget=self._settings.context_window.backward_budget,
                    forward_budget=self._settings.context_window.forward_budget,
                    target_budget=0,
                )
                member.window = SourceBlockWindow(
                    context_before=window.context_before,
                    target=window.target,
                    context_after=window.context_after,
                )
        return {
            'source_entity_hub_members': members,
            'source_entity_hub_groups': [[member] for member in members],
            'source_entity_hub_decision_cache': {},
        }

    async def load_candidates(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Retrieve current group proposals and expand them to all cross-pairs."""
        groups = state.source_entity_hub_groups
        proposals = await self._repository.read_source_entity_hub_candidates(
            state.source_uuid,
            groups,
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        groups_by_id = {
            min(member.uuid for member in group): group for group in groups
        }
        candidates = []
        for proposal in proposals:
            left_group = groups_by_id[proposal.left_group_uuid]
            right_group = groups_by_id[proposal.right_group_uuid]
            for left in left_group:
                for right in right_group:
                    candidates.append(_candidate(proposal, left, right))
        return {
            'source_entity_hub_candidates': candidates,
            'source_entity_hub_reranked_candidates': [],
        }

    def dispatch_rerank(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_hub_rerank_collect']:
        """Dispatch per left item; scores from separate requests are
        incomparable."""
        grouped: dict[str, list[SourceEntityHubCandidate]] = {}
        for candidate in state.source_entity_hub_candidates:
            if (
                _pair_key(candidate.left_uuid, candidate.right_uuid)
                not in state.source_entity_hub_decision_cache
            ):
                grouped.setdefault(candidate.left_uuid, []).append(candidate)
        if not grouped:
            return 'source_entity_hub_rerank_collect'
        return [
            Send(
                'source_entity_hub_rerank_worker',
                {
                    'source_entity_hub_rerank_ordinal': ordinal,
                    'source_entity_hub_left_text': _left_text(candidates[0]),
                    'source_entity_hub_candidates': candidates,
                },
            )
            for ordinal, candidates in enumerate(grouped.values())
        ]

    async def rerank_worker(
        self, state: SourceEntityHubRerankWorkerState
    ) -> dict[str, list[SourceEntityHubRerankResult]]:
        """Reject weak candidates without approving any identity pair."""
        candidates = state['source_entity_hub_candidates']
        results = await self._reranker.rerank(
            MultimodalContent(text=state['source_entity_hub_left_text']),
            [
                MultimodalContent(text=_right_text(candidate))
                for candidate in candidates
            ],
            content_images=[],
            top_n=None,
        )
        survivors = [
            candidates[result['index']]
            for result in sorted(results, key=lambda item: item['index'])
            if result['relevance_score']
            >= self._settings.reranker_rejection_threshold
        ]
        rejected = [
            candidate for candidate in candidates if candidate not in survivors
        ]
        return {
            'source_entity_hub_rerank_results': [
                SourceEntityHubRerankResult(
                    ordinal=state['source_entity_hub_rerank_ordinal'],
                    survivors=survivors,
                    rejected=rejected,
                )
            ]
        }

    def collect_rerank(self, state: SourceSemanticState) -> dict[str, object]:
        """Cache reranker rejections and clear reducer-backed round outputs."""
        results = sorted(
            state.source_entity_hub_rerank_results,
            key=lambda item: item.ordinal,
        )
        cache = dict(state.source_entity_hub_decision_cache)
        survivors = []
        for result in results:
            survivors.extend(result.survivors)
            for candidate in result.rejected:
                cache[_pair_key(candidate.left_uuid, candidate.right_uuid)] = (
                    False
                )
        return {
            'source_entity_hub_reranked_candidates': survivors,
            'source_entity_hub_decision_cache': cache,
            'source_entity_hub_rerank_results': Overwrite([]),
        }

    async def dispatch_judge(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_hub_judge_collect']:
        """Pack survivor pairs using call-local text and image budgets."""
        items = state.source_entity_hub_reranked_candidates
        if not items:
            return 'source_entity_hub_judge_collect'
        incoming = [
            items[index : index + self._settings.judge_batch_size]
            for index in range(0, len(items), self._settings.judge_batch_size)
        ]
        batches = await pack_content_batches(
            incoming,
            project=_project_judges,
            budget=self._judge_budget,
            image_budget=self._image_budget,
        )
        return [
            Send(
                'source_entity_hub_judge_worker',
                {
                    'source_entity_hub_judge_ordinal': ordinal,
                    'source_entity_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: SourceEntityHubJudgeWorkerState
    ) -> dict[str, list[SourceEntityHubJudgeResult]]:
        """Judge one bounded image catalog and preserve candidate identity."""
        batch = state['source_entity_hub_batch']
        builder = MultimodalContentBuilder()
        request_payload = _project_judges(batch, builder)
        payloads = await load_content_images(
            builder.images, budget=self._image_budget
        )
        decisions = await self._judge_module.acall(
            **request_payload,
            content_images=[dspy.Image(payload) for payload in payloads],
        )
        accepted = [
            batch[item.index] for item in decisions if item.belongs_in_same_hub
        ]
        rejected = [
            batch[item.index]
            for item in decisions
            if not item.belongs_in_same_hub
        ]
        return {
            'source_entity_hub_judge_results': [
                SourceEntityHubJudgeResult(
                    ordinal=state['source_entity_hub_judge_ordinal'],
                    accepted=accepted,
                    rejected=rejected,
                )
            ]
        }

    def collect_judge(self, state: SourceSemanticState) -> dict[str, object]:
        """Cache only explicit identity decisions and clear round worker outputs."""
        results = sorted(
            state.source_entity_hub_judge_results,
            key=lambda item: item.ordinal,
        )
        cache = dict(state.source_entity_hub_decision_cache)
        for result in results:
            for candidate in result.accepted:
                cache[_pair_key(candidate.left_uuid, candidate.right_uuid)] = (
                    True
                )
            for candidate in result.rejected:
                cache[_pair_key(candidate.left_uuid, candidate.right_uuid)] = (
                    False
                )
        return {
            'source_entity_hub_decision_cache': cache,
            'source_entity_hub_judge_results': Overwrite([]),
        }

    def merge_groups(self, state: SourceSemanticState) -> dict[str, object]:
        """Apply deterministic disjoint merges requiring every cross-pair positive."""
        groups = sorted(
            state.source_entity_hub_groups,
            key=lambda group: min(member.uuid for member in group),
        )
        group_by_id = {
            min(member.uuid for member in group): group for group in groups
        }
        proposals = sorted(
            state.source_entity_hub_candidates,
            key=lambda item: (
                -item.score,
                item.left_group_uuid,
                item.right_group_uuid,
            ),
        )
        consumed: set[str] = set()
        merged = []
        for proposal in proposals:
            left_id, right_id = (
                proposal.left_group_uuid,
                proposal.right_group_uuid,
            )
            if left_id in consumed or right_id in consumed:
                continue
            left, right = group_by_id[left_id], group_by_id[right_id]
            if all(
                state.source_entity_hub_decision_cache.get(
                    _pair_key(left_member.uuid, right_member.uuid)
                )
                is True
                for left_member in left
                for right_member in right
            ):
                consumed.update((left_id, right_id))
                merged.append(
                    sorted([*left, *right], key=lambda member: member.uuid)
                )
        next_groups = [
            group
            for group in groups
            if min(member.uuid for member in group) not in consumed
        ] + merged
        next_groups.sort(key=lambda group: min(member.uuid for member in group))
        return {
            'source_entity_hub_groups': next_groups,
            'source_entity_hub_round_merged': len(next_groups) < len(groups),
        }

    def repeat_round(self, state: SourceSemanticState) -> str:
        """Continue discovery only after at least one deterministic merge."""
        return (
            'source_entity_hub_load'
            if state.source_entity_hub_round_merged
            else 'source_entity_hub_prepare'
        )

    def prepare_hubs(self, state: SourceSemanticState) -> dict[str, object]:
        """Prepare identity-only hubs and stable, disjoint memberships."""
        groups = sorted(
            state.source_entity_hub_groups,
            key=lambda group: min(member.uuid for member in group),
        )
        return {
            'source_entity_hubs': [SourceEntityHub() for _ in groups],
            'source_entity_hub_memberships': [
                sorted(member.uuid for member in group) for group in groups
            ],
        }


def _candidate(
    proposal: SourceEntityHubCandidate,
    left: SourceEntityHubMember,
    right: SourceEntityHubMember,
) -> SourceEntityHubCandidate:
    return SourceEntityHubCandidate(
        left_group_uuid=proposal.left_group_uuid,
        right_group_uuid=proposal.right_group_uuid,
        left_uuid=left.uuid,
        left_name=left.name,
        left_description=left.description,
        left_window=left.window,
        right_uuid=right.uuid,
        right_name=right.name,
        right_description=right.description,
        right_window=right.window,
        score=proposal.score,
    )


def _pair_key(left_uuid: str, right_uuid: str) -> str:
    """Return a stable unordered occurrence-pair key."""
    left, right = sorted((left_uuid, right_uuid))
    return f'{left}:{right}'


def _left_text(candidate: SourceEntityHubCandidate) -> str:
    """Render the rerank query using the established name-description evidence."""
    return f'{candidate.left_name}: {candidate.left_description}'


def _right_text(candidate: SourceEntityHubCandidate) -> str:
    """Render a rerank document using the established name-description evidence."""
    return f'{candidate.right_name}: {candidate.right_description}'


def _judge_input(
    index: int,
    candidate: SourceEntityHubCandidate,
    builder: MultimodalContentBuilder,
) -> SourceEntityHubJudgeInput:
    """Project both occurrence windows into this judge call's shared catalog."""
    return SourceEntityHubJudgeInput(
        index=index,
        left_name=candidate.left_name,
        left_description=candidate.left_description,
        left_evidence=SourceEvidenceInput.from_window(
            candidate.left_window, builder
        ),
        right_name=candidate.right_name,
        right_description=candidate.right_description,
        right_evidence=SourceEvidenceInput.from_window(
            candidate.right_window, builder
        ),
    )


def _project_judges(
    candidates: list[SourceEntityHubCandidate],
    builder: MultimodalContentBuilder,
) -> dict[str, object]:
    """Project one ordered pair batch with request-local image positions."""
    return {
        'requests': [
            _judge_input(index, candidate, builder)
            for index, candidate in enumerate(candidates)
        ]
    }
