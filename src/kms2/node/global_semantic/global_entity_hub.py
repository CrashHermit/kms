"""Dispatch and collect global entity hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.global_semantic import GlobalEntityHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.global_semantic.global_entity_hub import (
    GlobalEntityHub,
    GlobalEntityHubCandidate,
    GlobalEntityHubJudgeInput,
    GlobalEntityHubJudgeResult,
    GlobalEntityHubMember,
    GlobalEntityHubRerankResult,
    GlobalEntityHubSummary,
    GlobalEntityHubSummaryInput,
    GlobalEntityHubSummaryMergeInput,
    GlobalEntityHubSummarySynthesisInput,
    GlobalEntityHubSynthesisInput,
    GlobalEntityHubSynthesisMember,
    GlobalEntityHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.global_semantic.global_entity_hub_repository import (
    GlobalEntityHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_entity_hub import (
    GlobalEntityHubModule,
    GlobalEntityHubSummaryMergeModule,
    GlobalEntityHubSummaryModule,
)
from kms2.module.global_semantic.global_entity_hub_judge import (
    GlobalEntityHubJudgeModule,
)


class GlobalEntityHubRerankWorkerState(TypedDict):
    """State supplied to one entity reranker worker."""

    global_entity_hub_rerank_ordinal: int
    global_entity_hub_left_text: str
    global_entity_hub_batch: list[GlobalEntityHubCandidate]


class GlobalEntityHubJudgeWorkerState(TypedDict):
    """State supplied to one entity judge worker."""

    global_entity_hub_judge_ordinal: int
    global_entity_hub_batch: list[GlobalEntityHubCandidate]


class GlobalEntityHubSynthesisWorkerState(TypedDict):
    """State supplied to one entity synthesis worker."""

    global_entity_hub_synthesis_ordinal: int
    global_entity_hub_community: list[GlobalEntityHubMember]


class GlobalEntityHubNode:
    """Run entity hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: GlobalEntityHubRepository,
        module: GlobalEntityHubModule,
        judge_module: GlobalEntityHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: GlobalEntityHubSettings,
        *,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
        summary_module: GlobalEntityHubSummaryModule,
        merge_module: GlobalEntityHubSummaryMergeModule,
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
        """Read source entity hub vector candidates."""
        candidates = await self._repository.read_global_entity_hub_candidates(
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'global_entity_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_entity_hub_rerank_collect']:
        """Partition entity candidates into ordered reranker batches."""
        if not state.global_entity_hub_candidates:
            return 'global_entity_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[GlobalEntityHubCandidate]] = {}
        for candidate in state.global_entity_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = _entity_text(
                left.left_canonical_name,
                left.left_description,
            )
            right_texts = [
                _entity_text(
                    candidate.right_canonical_name,
                    candidate.right_description,
                )
                for candidate in group
            ]
            costs = self._reranker_token_counter.count_texts(
                [left_text, *right_texts]
            )
            left_cost, *right_costs = costs
            batch: list[GlobalEntityHubCandidate] = []
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
                            'global_entity_hub_rerank_worker',
                            {
                                'global_entity_hub_rerank_ordinal': ordinal,
                                'global_entity_hub_left_text': left_text,
                                'global_entity_hub_batch': batch,
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
                        'global_entity_hub_rerank_worker',
                        {
                            'global_entity_hub_rerank_ordinal': ordinal,
                            'global_entity_hub_left_text': left_text,
                            'global_entity_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: GlobalEntityHubRerankWorkerState
    ) -> dict[str, list[GlobalEntityHubRerankResult]]:
        """Rerank one entity candidate batch."""
        batch = state['global_entity_hub_batch']
        results = await self._reranker.rerank(
            state['global_entity_hub_left_text'],
            [
                _entity_text(
                    candidate.right_canonical_name,
                    candidate.right_description,
                )
                for candidate in batch
            ],
            top_n=None,
        )
        direct: list[GlobalEntityHubCandidate] = []
        borderline: list[GlobalEntityHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'global_entity_hub_rerank_results': [
                GlobalEntityHubRerankResult(
                    ordinal=state['global_entity_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: GlobalSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.global_entity_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'global_entity_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'global_entity_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    async def dispatch_judge(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_entity_hub_judge_collect']:
        """Partition borderline entity pairs into ordered judge batches."""
        if not state.global_entity_hub_borderline_pairs:
            return 'global_entity_hub_judge_collect'

        batches = pack_items(
            state.global_entity_hub_borderline_pairs,
            token_counts=self._judge_budget.counter.count_texts(
                [
                    _entity_judge_input(0, item).model_dump_json(
                        exclude={'index'}
                    )
                    for item in state.global_entity_hub_borderline_pairs
                ]
            ),
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'global_entity_hub_judge_worker',
                {
                    'global_entity_hub_judge_ordinal': ordinal,
                    'global_entity_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: GlobalEntityHubJudgeWorkerState
    ) -> dict[str, list[GlobalEntityHubJudgeResult]]:
        """Judge one entity borderline batch."""
        batch = state['global_entity_hub_batch']
        decisions = await self._judge_module.acall(
            requests=[
                _entity_judge_input(index, item)
                for index, item in enumerate(batch)
            ]
        )
        accepted = [
            batch[decision.index]
            for decision in decisions
            if decision.belongs_in_same_hub
        ]
        return {
            'global_entity_hub_judge_results': [
                GlobalEntityHubJudgeResult(
                    ordinal=state['global_entity_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: GlobalSemanticState) -> dict[str, object]:
        """Merge direct and judged entity pairs in stable order."""
        accepted = list(state.global_entity_hub_direct_pairs)
        for result in sorted(
            state.global_entity_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], GlobalEntityHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'global_entity_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect entity communities."""
        await self._repository.replace_global_entity_hub_accepted_edges(
            state.global_entity_hub_accepted_pairs
        )
        communities = await self._repository.detect_global_entity_hub_communities(
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'global_entity_hub_communities': communities}

    def dispatch_synthesis(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_entity_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered entity community."""
        if not state.global_entity_hub_communities:
            return 'global_entity_hub_synthesis_collect'
        sends = [
            Send(
                'global_entity_hub_synthesis_worker',
                {
                    'global_entity_hub_synthesis_ordinal': ordinal,
                    'global_entity_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.global_entity_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: GlobalEntityHubSynthesisWorkerState
    ) -> dict[str, list[GlobalEntityHubSynthesisResult]]:
        """Reduce entity evidence to one global definition."""
        community = state['global_entity_hub_community']
        original_request = GlobalEntityHubSynthesisInput(
            members=[
                GlobalEntityHubSynthesisMember(
                    canonical_name=member.canonical_name,
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
                    request=GlobalEntityHubSummaryInput(evidence=batch)
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
                final_request = GlobalEntityHubSummarySynthesisInput(
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
            'global_entity_hub_synthesis_results': [
                GlobalEntityHubSynthesisResult(
                    ordinal=state['global_entity_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                    aliases=[member.canonical_name for member in community],
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[GlobalEntityHubSummary]
    ) -> list[GlobalEntityHubSummary]:
        """Merge one fitting level while carrying a trailing singleton."""

        async def merge_batch_fits(
            batch: list[GlobalEntityHubSummary],
        ) -> bool:
            return fits_token_budget(
                token_count=self._merge_budget.counter.count_texts(
                    [
                        GlobalEntityHubSummaryMergeInput(
                            summaries=batch
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._merge_budget.token_limit,
            )

        merged: list[GlobalEntityHubSummary] = []
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
                        request=GlobalEntityHubSummaryMergeInput(
                            summaries=batch
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Collect entity definitions in community order."""
        return {
            'global_entity_hub_synthesis_results_ordered': sorted(
                state.global_entity_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: GlobalSemanticState) -> dict[str, object]:
        """Embed ordered entity definitions and construct typed hubs."""
        results = state.global_entity_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'global_entity_hubs': [],
                'global_entity_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.canonical_name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            GlobalEntityHub(
                canonical_name=definition.canonical_name,
                aliases=result.aliases,
                description=definition.description,
                embedding=vector,
            )
            for result, definition, vector in zip(
                results, definitions, vectors, strict=True
            )
        ]
        return {
            'global_entity_hubs': hubs,
            'global_entity_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _entity_text(canonical_name: str, description: str) -> str:
    """Render one canonical entity for reranking."""
    return f'{canonical_name}: {description}'


def _entity_judge_input(
    index: int, candidate: GlobalEntityHubCandidate
) -> GlobalEntityHubJudgeInput:
    """Build one indexed entity judge request."""
    return GlobalEntityHubJudgeInput(
        index=index,
        left_canonical_name=candidate.left_canonical_name,
        left_description=candidate.left_description,
        right_canonical_name=candidate.right_canonical_name,
        right_description=candidate.right_description,
    )
