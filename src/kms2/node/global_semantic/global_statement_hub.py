"""Dispatch and collect global statement hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.global_semantic import GlobalStatementHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.global_semantic.global_statement_hub import (
    GlobalStatementHub,
    GlobalStatementHubCandidate,
    GlobalStatementHubJudgeInput,
    GlobalStatementHubJudgeResult,
    GlobalStatementHubMember,
    GlobalStatementHubRerankResult,
    GlobalStatementHubSummary,
    GlobalStatementHubSummaryInput,
    GlobalStatementHubSummaryMergeInput,
    GlobalStatementHubSummarySynthesisInput,
    GlobalStatementHubSynthesisInput,
    GlobalStatementHubSynthesisMember,
    GlobalStatementHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.global_semantic.global_statement_hub_repository import (
    GlobalStatementHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_statement_hub import (
    GlobalStatementHubModule,
    GlobalStatementHubSummaryMergeModule,
    GlobalStatementHubSummaryModule,
)
from kms2.module.global_semantic.global_statement_hub_judge import (
    GlobalStatementHubJudgeModule,
)


class GlobalStatementHubRerankWorkerState(TypedDict):
    """State supplied to one statement reranker worker."""

    global_statement_hub_rerank_ordinal: int
    global_statement_hub_left_text: str
    global_statement_hub_batch: list[GlobalStatementHubCandidate]


class GlobalStatementHubJudgeWorkerState(TypedDict):
    """State supplied to one statement judge worker."""

    global_statement_hub_judge_ordinal: int
    global_statement_hub_batch: list[GlobalStatementHubCandidate]


class GlobalStatementHubSynthesisWorkerState(TypedDict):
    """State supplied to one statement synthesis worker."""

    global_statement_hub_synthesis_ordinal: int
    global_statement_hub_community: list[GlobalStatementHubMember]


class GlobalStatementHubNode:
    """Run statement hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: GlobalStatementHubRepository,
        module: GlobalStatementHubModule,
        judge_module: GlobalStatementHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: GlobalStatementHubSettings,
        *,
        summary_module: GlobalStatementHubSummaryModule,
        merge_module: GlobalStatementHubSummaryMergeModule,
        final_budget: TokenBudget,
        summary_budget: TokenBudget,
        merge_budget: TokenBudget,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
    ) -> None:
        self._repository = repository
        self._module = module
        self._summary_module = summary_module
        self._merge_module = merge_module
        self._final_budget = final_budget
        self._summary_budget = summary_budget
        self._merge_budget = merge_budget
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings
        self._reranker_token_counter = reranker_token_counter
        self._reranker_overhead_tokens = reranker_overhead_tokens
        self._judge_budget = judge_budget

    async def load_candidates(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Read source statement hub vector candidates."""
        candidates = (
            await self._repository.read_global_statement_hub_candidates(
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        return {'global_statement_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_statement_hub_rerank_collect']:
        """Partition statement candidates into ordered reranker batches."""
        if not state.global_statement_hub_candidates:
            return 'global_statement_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[GlobalStatementHubCandidate]] = {}
        for candidate in state.global_statement_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = _statement_text(left.left_description)
            right_texts = [
                _statement_text(candidate.right_description)
                for candidate in group
            ]
            costs = self._reranker_token_counter.count_texts(
                [left_text, *right_texts]
            )
            left_cost, *right_costs = costs
            batch: list[GlobalStatementHubCandidate] = []
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
                            'global_statement_hub_rerank_worker',
                            {
                                'global_statement_hub_rerank_ordinal': ordinal,
                                'global_statement_hub_left_text': left_text,
                                'global_statement_hub_batch': batch,
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
                        'global_statement_hub_rerank_worker',
                        {
                            'global_statement_hub_rerank_ordinal': ordinal,
                            'global_statement_hub_left_text': left_text,
                            'global_statement_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: GlobalStatementHubRerankWorkerState
    ) -> dict[str, list[GlobalStatementHubRerankResult]]:
        """Rerank one statement candidate batch."""
        batch = state['global_statement_hub_batch']
        results = await self._reranker.rerank(
            state['global_statement_hub_left_text'],
            [
                _statement_text(candidate.right_description)
                for candidate in batch
            ],
            top_n=None,
        )
        direct: list[GlobalStatementHubCandidate] = []
        borderline: list[GlobalStatementHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'global_statement_hub_rerank_results': [
                GlobalStatementHubRerankResult(
                    ordinal=state['global_statement_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: GlobalSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.global_statement_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'global_statement_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'global_statement_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    async def dispatch_judge(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_statement_hub_judge_collect']:
        """Partition borderline statement pairs into ordered judge batches."""
        if not state.global_statement_hub_borderline_pairs:
            return 'global_statement_hub_judge_collect'

        batches = pack_items(
            state.global_statement_hub_borderline_pairs,
            token_counts=self._judge_budget.counter.count_texts(
                [
                    _statement_judge_input(0, item).model_dump_json(
                        exclude={'index'}
                    )
                    for item in state.global_statement_hub_borderline_pairs
                ]
            ),
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'global_statement_hub_judge_worker',
                {
                    'global_statement_hub_judge_ordinal': ordinal,
                    'global_statement_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: GlobalStatementHubJudgeWorkerState
    ) -> dict[str, list[GlobalStatementHubJudgeResult]]:
        """Judge one statement borderline batch."""
        batch = state['global_statement_hub_batch']
        decisions = await self._judge_module.acall(
            requests=[
                _statement_judge_input(index, item)
                for index, item in enumerate(batch)
            ]
        )
        accepted = [
            batch[decision.index]
            for decision in decisions
            if decision.belongs_in_same_hub
        ]
        return {
            'global_statement_hub_judge_results': [
                GlobalStatementHubJudgeResult(
                    ordinal=state['global_statement_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: GlobalSemanticState) -> dict[str, object]:
        """Merge direct and judged statement pairs in stable order."""
        accepted = list(state.global_statement_hub_direct_pairs)
        for result in sorted(
            state.global_statement_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], GlobalStatementHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'global_statement_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect statement communities."""
        await self._repository.replace_global_statement_hub_accepted_edges(
            state.global_statement_hub_accepted_pairs
        )
        communities = await self._repository.detect_global_statement_hub_communities(
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'global_statement_hub_communities': communities}

    def dispatch_synthesis(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_statement_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered statement community."""
        if not state.global_statement_hub_communities:
            return 'global_statement_hub_synthesis_collect'
        sends = [
            Send(
                'global_statement_hub_synthesis_worker',
                {
                    'global_statement_hub_synthesis_ordinal': ordinal,
                    'global_statement_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.global_statement_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: GlobalStatementHubSynthesisWorkerState
    ) -> dict[str, list[GlobalStatementHubSynthesisResult]]:
        """Synthesize one statement community through evidence reduction."""
        community = state['global_statement_hub_community']
        request = GlobalStatementHubSynthesisInput(
            members=[
                GlobalStatementHubSynthesisMember(
                    description=member.description
                )
                for member in community
            ]
        )
        if fits_token_budget(
            token_count=self._final_budget.counter.count_texts(
                [request.model_dump_json()]
            )[0],
            threshold=self._final_budget.token_limit,
        ):
            definition = await self._module.acall(request=request)
        else:
            records = [member.model_dump_json() for member in request.members]
            batches = pack_items(
                records,
                token_counts=self._summary_budget.counter.count_texts(records),
                token_budget=self._summary_budget.token_limit,
            )
            current = [
                await self._summary_module.acall(
                    request=GlobalStatementHubSummaryInput(evidence=batch)
                )
                for batch in batches
            ]
            while True:
                final_request = GlobalStatementHubSummarySynthesisInput(
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
            'global_statement_hub_synthesis_results': [
                GlobalStatementHubSynthesisResult(
                    ordinal=state['global_statement_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                    aliases=[member.description for member in community],
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[GlobalStatementHubSummary]
    ) -> list[GlobalStatementHubSummary]:
        batches = pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        )
        merged: list[GlobalStatementHubSummary] = []
        for batch in batches:
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=GlobalStatementHubSummaryMergeInput(
                            summaries=batch
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Collect statement definitions in community order."""
        return {
            'global_statement_hub_synthesis_results_ordered': sorted(
                state.global_statement_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: GlobalSemanticState) -> dict[str, object]:
        """Embed ordered statement definitions and construct typed hubs."""
        results = state.global_statement_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'global_statement_hubs': [],
                'global_statement_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [definition.description for definition in definitions]
        )
        hubs = [
            GlobalStatementHub(
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
            'global_statement_hubs': hubs,
            'global_statement_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _statement_text(description: str) -> str:
    """Render one statement description for reranking."""
    return description


def _statement_judge_input(
    index: int, candidate: GlobalStatementHubCandidate
) -> GlobalStatementHubJudgeInput:
    """Build one indexed statement judge request."""
    return GlobalStatementHubJudgeInput(
        index=index,
        left_description=candidate.left_description,
        right_description=candidate.right_description,
    )
