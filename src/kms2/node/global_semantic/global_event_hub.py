"""Dispatch and collect global event hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.global_semantic import GlobalEventHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.global_semantic.global_event_hub import (
    GlobalEventHub,
    GlobalEventHubCandidate,
    GlobalEventHubJudgeInput,
    GlobalEventHubJudgeResult,
    GlobalEventHubMember,
    GlobalEventHubRerankResult,
    GlobalEventHubSummary,
    GlobalEventHubSummaryInput,
    GlobalEventHubSummaryMergeInput,
    GlobalEventHubSummarySynthesisInput,
    GlobalEventHubSynthesisInput,
    GlobalEventHubSynthesisMember,
    GlobalEventHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.global_semantic.global_event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_event_hub import (
    GlobalEventHubModule,
    GlobalEventHubSummaryMergeModule,
    GlobalEventHubSummaryModule,
)
from kms2.module.global_semantic.global_event_hub_judge import (
    GlobalEventHubJudgeModule,
)


class GlobalEventHubRerankWorkerState(TypedDict):
    """State supplied to one event reranker worker."""

    global_event_hub_rerank_ordinal: int
    global_event_hub_left_text: str
    global_event_hub_batch: list[GlobalEventHubCandidate]


class GlobalEventHubJudgeWorkerState(TypedDict):
    """State supplied to one event judge worker."""

    global_event_hub_judge_ordinal: int
    global_event_hub_batch: list[GlobalEventHubCandidate]


class GlobalEventHubSynthesisWorkerState(TypedDict):
    """State supplied to one event synthesis worker."""

    global_event_hub_synthesis_ordinal: int
    global_event_hub_community: list[GlobalEventHubMember]


class GlobalEventHubNode:
    """Run event hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: GlobalEventHubRepository,
        module: GlobalEventHubModule,
        judge_module: GlobalEventHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: GlobalEventHubSettings,
        *,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
        summary_module: GlobalEventHubSummaryModule,
        merge_module: GlobalEventHubSummaryMergeModule,
        final_budget: TokenBudget,
        summary_budget: TokenBudget,
        merge_budget: TokenBudget,
    ) -> None:
        self._repository = repository
        self._module = module
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings
        self._summary_module = summary_module
        self._merge_module = merge_module
        self._final_budget = final_budget
        self._summary_budget = summary_budget
        self._merge_budget = merge_budget
        self._reranker_token_counter = reranker_token_counter
        self._reranker_overhead_tokens = reranker_overhead_tokens
        self._judge_budget = judge_budget

    async def load_candidates(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Read source event hub vector candidates."""
        candidates = await self._repository.read_global_event_hub_candidates(
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'global_event_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_event_hub_rerank_collect']:
        """Partition event candidates into ordered reranker batches."""
        if not state.global_event_hub_candidates:
            return 'global_event_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[GlobalEventHubCandidate]] = {}
        for candidate in state.global_event_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = _event_text(left.left_name, left.left_description)
            right_texts = [
                _event_text(candidate.right_name, candidate.right_description)
                for candidate in group
            ]
            costs = self._reranker_token_counter.count_texts(
                [left_text, *right_texts]
            )
            left_cost, *right_costs = costs
            batch: list[GlobalEventHubCandidate] = []
            batch_cost = 0
            for candidate, right_cost in zip(group, right_costs, strict=True):
                if (
                    batch
                    and batch_cost
                    + left_cost
                    + right_cost
                    + self._reranker_overhead_tokens
                    > self._settings.reranker_token_budget
                ):
                    sends.append(
                        Send(
                            'global_event_hub_rerank_worker',
                            {
                                'global_event_hub_rerank_ordinal': ordinal,
                                'global_event_hub_left_text': left_text,
                                'global_event_hub_batch': batch,
                            },
                        )
                    )
                    ordinal += 1
                    batch = []
                    batch_cost = 0
                batch.append(candidate)
                batch_cost += (
                    left_cost + right_cost + self._reranker_overhead_tokens
                )
            if batch:
                sends.append(
                    Send(
                        'global_event_hub_rerank_worker',
                        {
                            'global_event_hub_rerank_ordinal': ordinal,
                            'global_event_hub_left_text': left_text,
                            'global_event_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: GlobalEventHubRerankWorkerState
    ) -> dict[str, list[GlobalEventHubRerankResult]]:
        """Rerank one event candidate batch."""
        batch = state['global_event_hub_batch']
        results = await self._reranker.rerank(
            state['global_event_hub_left_text'],
            [
                _event_text(
                    candidate.right_name,
                    candidate.right_description,
                )
                for candidate in batch
            ],
            top_n=None,
        )
        direct: list[GlobalEventHubCandidate] = []
        borderline: list[GlobalEventHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'global_event_hub_rerank_results': [
                GlobalEventHubRerankResult(
                    ordinal=state['global_event_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: GlobalSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.global_event_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'global_event_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'global_event_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    async def dispatch_judge(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_event_hub_judge_collect']:
        """Partition borderline event pairs into ordered judge batches."""
        if not state.global_event_hub_borderline_pairs:
            return 'global_event_hub_judge_collect'

        batches = pack_items(
            state.global_event_hub_borderline_pairs,
            token_counts=self._judge_budget.counter.count_texts(
                [
                    _event_judge_input(0, item).model_dump_json(
                        exclude={'index'}
                    )
                    for item in state.global_event_hub_borderline_pairs
                ]
            ),
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'global_event_hub_judge_worker',
                {
                    'global_event_hub_judge_ordinal': ordinal,
                    'global_event_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: GlobalEventHubJudgeWorkerState
    ) -> dict[str, list[GlobalEventHubJudgeResult]]:
        """Judge one event borderline batch."""
        batch = state['global_event_hub_batch']
        decisions = await self._judge_module.acall(
            requests=[
                _event_judge_input(index, item)
                for index, item in enumerate(batch)
            ]
        )
        accepted = [
            batch[decision.index]
            for decision in decisions
            if decision.belongs_in_same_hub
        ]
        return {
            'global_event_hub_judge_results': [
                GlobalEventHubJudgeResult(
                    ordinal=state['global_event_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: GlobalSemanticState) -> dict[str, object]:
        """Merge direct and judged event pairs in stable order."""
        accepted = list(state.global_event_hub_direct_pairs)
        for result in sorted(
            state.global_event_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], GlobalEventHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'global_event_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect event communities."""
        await self._repository.replace_global_event_hub_accepted_edges(
            state.global_event_hub_accepted_pairs
        )
        communities = await self._repository.detect_global_event_hub_communities(
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'global_event_hub_communities': communities}

    def dispatch_synthesis(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_event_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered event community."""
        if not state.global_event_hub_communities:
            return 'global_event_hub_synthesis_collect'
        sends = [
            Send(
                'global_event_hub_synthesis_worker',
                {
                    'global_event_hub_synthesis_ordinal': ordinal,
                    'global_event_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.global_event_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: GlobalEventHubSynthesisWorkerState
    ) -> dict[str, list[GlobalEventHubSynthesisResult]]:
        """Reduce event evidence to one global definition."""
        community = state['global_event_hub_community']
        original_request = GlobalEventHubSynthesisInput(
            members=[
                GlobalEventHubSynthesisMember(
                    name=member.name,
                    description=member.description,
                )
                for member in community
            ]
        )
        if fits_token_budget(
            token_count=self._final_budget.counter.count_texts(
                [original_request.model_dump_json()]
            )[0],
            threshold=self._final_budget.token_limit,
        ):
            definition = await self._module.acall(request=original_request)
        else:
            records = [
                member.model_dump_json() for member in original_request.members
            ]

            current = [
                await self._summary_module.acall(
                    request=GlobalEventHubSummaryInput(evidence=batch)
                )
                for batch in pack_items(
                    records,
                    token_counts=self._summary_budget.counter.count_texts(
                        records
                    ),
                    token_budget=self._summary_budget.token_limit,
                )
            ]
            while True:
                final_request = GlobalEventHubSummarySynthesisInput(
                    summaries=current
                )
                if fits_token_budget(
                    token_count=self._final_budget.counter.count_texts(
                        [final_request.model_dump_json()]
                    )[0],
                    threshold=self._final_budget.token_limit,
                ):
                    definition = await self._module.acall(request=final_request)
                    break
                current = await self._merge_summary_level(current)
        return {
            'global_event_hub_synthesis_results': [
                GlobalEventHubSynthesisResult(
                    ordinal=state['global_event_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                    aliases=[member.name for member in community],
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[GlobalEventHubSummary]
    ) -> list[GlobalEventHubSummary]:
        """Merge one fitting level while carrying a trailing singleton."""

        async def merge_batch_fits(
            batch: list[GlobalEventHubSummary],
        ) -> bool:
            return fits_token_budget(
                token_count=self._merge_budget.counter.count_texts(
                    [
                        GlobalEventHubSummaryMergeInput(
                            summaries=batch
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._merge_budget.token_limit,
            )

        merged: list[GlobalEventHubSummary] = []
        for batch in pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        ):
            if len(batch) == 1:
                merged.extend(batch)
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=GlobalEventHubSummaryMergeInput(summaries=batch)
                    )
                )
        return merged

    def collect_synthesis(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Collect event definitions in community order."""
        return {
            'global_event_hub_synthesis_results_ordered': sorted(
                state.global_event_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: GlobalSemanticState) -> dict[str, object]:
        """Embed ordered event definitions and construct typed hubs."""
        results = state.global_event_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'global_event_hubs': [],
                'global_event_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            GlobalEventHub(
                name=definition.name,
                aliases=result.aliases,
                description=definition.description,
                embedding=vector,
            )
            for result, definition, vector in zip(
                results, definitions, vectors, strict=True
            )
        ]
        return {
            'global_event_hubs': hubs,
            'global_event_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _event_text(name: str, description: str) -> str:
    """Render one canonical event for reranking."""
    return f'{name}: {description}'


def _event_judge_input(
    index: int, candidate: GlobalEventHubCandidate
) -> GlobalEventHubJudgeInput:
    """Build one indexed event judge request."""
    return GlobalEventHubJudgeInput(
        index=index,
        left_name=candidate.left_name,
        left_description=candidate.left_description,
        right_name=candidate.right_name,
        right_description=candidate.right_description,
    )
