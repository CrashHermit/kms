"""Dispatch and collect source-local statement hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.source_semantic import SourceStatementHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_semantic.source_statement_hub import (
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubJudgeInput,
    SourceStatementHubJudgeResult,
    SourceStatementHubMember,
    SourceStatementHubRerankResult,
    SourceStatementHubSummary,
    SourceStatementHubSummaryInput,
    SourceStatementHubSummaryMergeInput,
    SourceStatementHubSummarySynthesisInput,
    SourceStatementHubSynthesisInput,
    SourceStatementHubSynthesisMember,
    SourceStatementHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_statement_hub import (
    SourceStatementHubModule,
    SourceStatementHubSummaryMergeModule,
    SourceStatementHubSummaryModule,
)
from kms2.module.source_semantic.source_statement_hub_judge import (
    SourceStatementHubJudgeModule,
)


class SourceStatementHubRerankWorkerState(TypedDict):
    """State supplied to one statement reranker worker."""

    source_statement_hub_rerank_ordinal: int
    source_statement_hub_left_text: str
    source_statement_hub_batch: list[SourceStatementHubCandidate]


class SourceStatementHubJudgeWorkerState(TypedDict):
    """State supplied to one statement judge worker."""

    source_statement_hub_judge_ordinal: int
    source_statement_hub_batch: list[SourceStatementHubCandidate]


class SourceStatementHubSynthesisWorkerState(TypedDict):
    """State supplied to one statement synthesis worker."""

    source_statement_hub_synthesis_ordinal: int
    source_statement_hub_community: list[SourceStatementHubMember]


class SourceStatementHubNode:
    """Run statement hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: SourceStatementRepository,
        module: SourceStatementHubModule,
        judge_module: SourceStatementHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: SourceStatementHubSettings,
        *,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
        summary_module: SourceStatementHubSummaryModule,
        merge_module: SourceStatementHubSummaryMergeModule,
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
        self._reranker_token_counter = reranker_token_counter
        self._reranker_overhead_tokens = reranker_overhead_tokens
        self._judge_budget = judge_budget
        self._summary_module = summary_module
        self._merge_module = merge_module
        self._final_budget = final_budget
        self._summary_budget = summary_budget
        self._merge_budget = merge_budget

    async def load_candidates(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Read statement vector candidates for this source."""
        candidates = (
            await self._repository.read_source_statement_hub_candidates(
                state.source_uuid,
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        return {'source_statement_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_statement_hub_rerank_collect']:
        """Partition statement candidates into ordered reranker batches."""
        if not state.source_statement_hub_candidates:
            return 'source_statement_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[SourceStatementHubCandidate]] = {}
        for candidate in state.source_statement_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = left.left_description
            right_texts = [candidate.right_description for candidate in group]
            costs = self._reranker_token_counter.count_texts(
                [left_text, *right_texts]
            )
            left_cost, *right_costs = costs
            batch: list[SourceStatementHubCandidate] = []
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
                            'source_statement_hub_rerank_worker',
                            {
                                'source_statement_hub_rerank_ordinal': ordinal,
                                'source_statement_hub_left_text': left_text,
                                'source_statement_hub_batch': batch,
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
                        'source_statement_hub_rerank_worker',
                        {
                            'source_statement_hub_rerank_ordinal': ordinal,
                            'source_statement_hub_left_text': left_text,
                            'source_statement_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: SourceStatementHubRerankWorkerState
    ) -> dict[str, list[SourceStatementHubRerankResult]]:
        """Rerank one statement candidate batch."""
        batch = state['source_statement_hub_batch']
        results = await self._reranker.rerank(
            state['source_statement_hub_left_text'],
            [candidate.right_description for candidate in batch],
            top_n=None,
        )
        direct: list[SourceStatementHubCandidate] = []
        borderline: list[SourceStatementHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'source_statement_hub_rerank_results': [
                SourceStatementHubRerankResult(
                    ordinal=state['source_statement_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: SourceSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.source_statement_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'source_statement_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'source_statement_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    async def dispatch_judge(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_statement_hub_judge_collect']:
        """Partition borderline statement pairs into ordered judge batches."""
        if not state.source_statement_hub_borderline_pairs:
            return 'source_statement_hub_judge_collect'

        batches = pack_items(
            state.source_statement_hub_borderline_pairs,
            token_counts=self._judge_budget.counter.count_texts(
                [
                    _statement_judge_input(0, item).model_dump_json(
                        exclude={'index'}
                    )
                    for item in state.source_statement_hub_borderline_pairs
                ]
            ),
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'source_statement_hub_judge_worker',
                {
                    'source_statement_hub_judge_ordinal': ordinal,
                    'source_statement_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: SourceStatementHubJudgeWorkerState
    ) -> dict[str, list[SourceStatementHubJudgeResult]]:
        """Judge one statement borderline batch."""
        batch = state['source_statement_hub_batch']
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
            'source_statement_hub_judge_results': [
                SourceStatementHubJudgeResult(
                    ordinal=state['source_statement_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: SourceSemanticState) -> dict[str, object]:
        """Merge direct and judged statement pairs in stable order."""
        accepted = list(state.source_statement_hub_direct_pairs)
        for result in sorted(
            state.source_statement_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], SourceStatementHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'source_statement_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect statement communities."""
        await self._repository.replace_source_statement_accepted_edges(
            state.source_uuid, state.source_statement_hub_accepted_pairs
        )
        communities = await self._repository.detect_source_statement_communities(
            state.source_uuid,
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'source_statement_hub_communities': communities}

    def dispatch_synthesis(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_statement_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered statement community."""
        if not state.source_statement_hub_communities:
            return 'source_statement_hub_synthesis_collect'
        sends = [
            Send(
                'source_statement_hub_synthesis_worker',
                {
                    'source_statement_hub_synthesis_ordinal': ordinal,
                    'source_statement_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.source_statement_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: SourceStatementHubSynthesisWorkerState
    ) -> dict[str, list[SourceStatementHubSynthesisResult]]:
        """Synthesize one statement community using evidence reduction."""
        community = state['source_statement_hub_community']
        synthesis_members = [
            SourceStatementHubSynthesisMember(description=member.description)
            for member in community
        ]
        request = SourceStatementHubSynthesisInput(members=synthesis_members)
        if fits_token_budget(
            token_count=self._final_budget.counter.count_texts(
                [request.model_dump_json()]
            )[0],
            threshold=self._final_budget.token_limit,
        ):
            definition = await self._module.acall(request=request)
        else:
            evidence = [
                member.model_dump_json() for member in synthesis_members
            ]

            current = [
                await self._summary_module.acall(
                    request=SourceStatementHubSummaryInput(evidence=batch)
                )
                for batch in pack_items(
                    evidence,
                    token_counts=self._summary_budget.counter.count_texts(
                        evidence
                    ),
                    token_budget=self._summary_budget.token_limit,
                )
            ]
            while not fits_token_budget(
                token_count=self._final_budget.counter.count_texts(
                    [
                        SourceStatementHubSummarySynthesisInput(
                            summaries=current
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._final_budget.token_limit,
            ):
                current = await self._merge_summary_level(current)
            definition = await self._module.acall(
                request=SourceStatementHubSummarySynthesisInput(
                    summaries=current
                )
            )
        return {
            'source_statement_hub_synthesis_results': [
                SourceStatementHubSynthesisResult(
                    ordinal=state['source_statement_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[SourceStatementHubSummary]
    ) -> list[SourceStatementHubSummary]:
        """Merge one level, carrying a trailing singleton unchanged."""

        async def merge_batch_fits(
            batch: list[SourceStatementHubSummary],
        ) -> bool:
            return fits_token_budget(
                token_count=self._merge_budget.counter.count_texts(
                    [
                        SourceStatementHubSummaryMergeInput(
                            summaries=batch
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._merge_budget.token_limit,
            )

        batches = pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        )
        return [
            await self._merge_module.acall(
                request=SourceStatementHubSummaryMergeInput(summaries=batch)
            )
            if len(batch) > 1
            else batch[0]
            for batch in batches
        ]

    def collect_synthesis(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Collect statement definitions in community order."""
        return {
            'source_statement_hub_synthesis_results_ordered': sorted(
                state.source_statement_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SourceSemanticState) -> dict[str, object]:
        """Embed ordered statement definitions and construct typed hubs."""
        results = state.source_statement_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'source_statement_hubs': [],
                'source_statement_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.canonical_name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            SourceStatementHub(
                source_uuid=state.source_uuid,
                canonical_name=definition.canonical_name,
                description=definition.description,
                embedding=vector,
            )
            for definition, vector in zip(definitions, vectors, strict=True)
        ]
        return {
            'source_statement_hubs': hubs,
            'source_statement_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _statement_judge_input(
    index: int, candidate: SourceStatementHubCandidate
) -> SourceStatementHubJudgeInput:
    """Build one indexed statement judge request."""
    return SourceStatementHubJudgeInput(
        index=index,
        left_description=candidate.left_description,
        right_description=candidate.right_description,
    )
