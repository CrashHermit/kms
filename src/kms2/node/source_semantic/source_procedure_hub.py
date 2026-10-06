"""Dispatch and collect source-local procedure hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.source_semantic import SourceProcedureHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHub,
    SourceProcedureHubCandidate,
    SourceProcedureHubJudgeInput,
    SourceProcedureHubJudgeResult,
    SourceProcedureHubMember,
    SourceProcedureHubRerankResult,
    SourceProcedureHubSummary,
    SourceProcedureHubSummaryInput,
    SourceProcedureHubSummaryMergeInput,
    SourceProcedureHubSummarySynthesisInput,
    SourceProcedureHubSynthesisInput,
    SourceProcedureHubSynthesisMember,
    SourceProcedureHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_procedure_hub import (
    SourceProcedureHubModule,
    SourceProcedureHubSummaryMergeModule,
    SourceProcedureHubSummaryModule,
)
from kms2.module.source_semantic.source_procedure_hub_judge import (
    SourceProcedureHubJudgeModule,
)


class SourceProcedureHubRerankWorkerState(TypedDict):
    """State supplied to one procedure reranker worker."""

    source_procedure_hub_rerank_ordinal: int
    source_procedure_hub_left_text: str
    source_procedure_hub_batch: list[SourceProcedureHubCandidate]


class SourceProcedureHubJudgeWorkerState(TypedDict):
    """State supplied to one procedure judge worker."""

    source_procedure_hub_judge_ordinal: int
    source_procedure_hub_batch: list[SourceProcedureHubCandidate]


class SourceProcedureHubSynthesisWorkerState(TypedDict):
    """State supplied to one procedure synthesis worker."""

    source_procedure_hub_synthesis_ordinal: int
    source_procedure_hub_community: list[SourceProcedureHubMember]


class SourceProcedureHubNode:
    """Run procedure hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: SourceProcedureRepository,
        module: SourceProcedureHubModule,
        judge_module: SourceProcedureHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: SourceProcedureHubSettings,
        *,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
        summary_module: SourceProcedureHubSummaryModule,
        merge_module: SourceProcedureHubSummaryMergeModule,
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
        """Read procedure vector candidates for this source."""
        candidates = (
            await self._repository.read_source_procedure_hub_candidates(
                state.source_uuid,
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        return {'source_procedure_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_procedure_hub_rerank_collect']:
        """Partition procedure candidates into ordered reranker batches."""
        if not state.source_procedure_hub_candidates:
            return 'source_procedure_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[SourceProcedureHubCandidate]] = {}
        for candidate in state.source_procedure_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = left.left_description
            right_texts = [candidate.right_description for candidate in group]
            costs = self._reranker_token_counter.count_texts(
                [left_text, *right_texts]
            )
            left_cost, *right_costs = costs
            batch: list[SourceProcedureHubCandidate] = []
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
                            'source_procedure_hub_rerank_worker',
                            {
                                'source_procedure_hub_rerank_ordinal': ordinal,
                                'source_procedure_hub_left_text': left_text,
                                'source_procedure_hub_batch': batch,
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
                        'source_procedure_hub_rerank_worker',
                        {
                            'source_procedure_hub_rerank_ordinal': ordinal,
                            'source_procedure_hub_left_text': left_text,
                            'source_procedure_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: SourceProcedureHubRerankWorkerState
    ) -> dict[str, list[SourceProcedureHubRerankResult]]:
        """Rerank one procedure candidate batch."""
        batch = state['source_procedure_hub_batch']
        results = await self._reranker.rerank(
            state['source_procedure_hub_left_text'],
            [candidate.right_description for candidate in batch],
            top_n=None,
        )
        direct: list[SourceProcedureHubCandidate] = []
        borderline: list[SourceProcedureHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'source_procedure_hub_rerank_results': [
                SourceProcedureHubRerankResult(
                    ordinal=state['source_procedure_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: SourceSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.source_procedure_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'source_procedure_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'source_procedure_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    async def dispatch_judge(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_procedure_hub_judge_collect']:
        """Partition borderline procedure pairs into ordered judge batches."""
        if not state.source_procedure_hub_borderline_pairs:
            return 'source_procedure_hub_judge_collect'

        batches = pack_items(
            state.source_procedure_hub_borderline_pairs,
            token_counts=self._judge_budget.counter.count_texts(
                [
                    _procedure_judge_input(0, item).model_dump_json(
                        exclude={'index'}
                    )
                    for item in state.source_procedure_hub_borderline_pairs
                ]
            ),
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'source_procedure_hub_judge_worker',
                {
                    'source_procedure_hub_judge_ordinal': ordinal,
                    'source_procedure_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: SourceProcedureHubJudgeWorkerState
    ) -> dict[str, list[SourceProcedureHubJudgeResult]]:
        """Judge one procedure borderline batch."""
        batch = state['source_procedure_hub_batch']
        decisions = await self._judge_module.acall(
            requests=[
                _procedure_judge_input(index, item)
                for index, item in enumerate(batch)
            ]
        )
        accepted = [
            batch[decision.index]
            for decision in decisions
            if decision.belongs_in_same_hub
        ]
        return {
            'source_procedure_hub_judge_results': [
                SourceProcedureHubJudgeResult(
                    ordinal=state['source_procedure_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: SourceSemanticState) -> dict[str, object]:
        """Merge direct and judged procedure pairs in stable order."""
        accepted = list(state.source_procedure_hub_direct_pairs)
        for result in sorted(
            state.source_procedure_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], SourceProcedureHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'source_procedure_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect procedure communities."""
        await self._repository.replace_source_procedure_accepted_edges(
            state.source_uuid, state.source_procedure_hub_accepted_pairs
        )
        communities = await self._repository.detect_source_procedure_communities(
            state.source_uuid,
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'source_procedure_hub_communities': communities}

    def dispatch_synthesis(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_procedure_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered procedure community."""
        if not state.source_procedure_hub_communities:
            return 'source_procedure_hub_synthesis_collect'
        sends = [
            Send(
                'source_procedure_hub_synthesis_worker',
                {
                    'source_procedure_hub_synthesis_ordinal': ordinal,
                    'source_procedure_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.source_procedure_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: SourceProcedureHubSynthesisWorkerState
    ) -> dict[str, list[SourceProcedureHubSynthesisResult]]:
        """Synthesize one procedure community."""
        community = state['source_procedure_hub_community']
        request = SourceProcedureHubSynthesisInput(
            members=[
                SourceProcedureHubSynthesisMember(
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
            evidence = [member.model_dump_json() for member in request.members]
            summaries = await self._summarize_evidence(evidence)
            while not fits_token_budget(
                token_count=self._final_budget.counter.count_texts(
                    [
                        SourceProcedureHubSummarySynthesisInput(
                            summaries=summaries
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._final_budget.token_limit,
            ):
                summaries = await self._merge_summary_level(summaries)
            definition = await self._module.acall(
                request=SourceProcedureHubSummarySynthesisInput(
                    summaries=summaries
                )
            )
        return {
            'source_procedure_hub_synthesis_results': [
                SourceProcedureHubSynthesisResult(
                    ordinal=state['source_procedure_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                )
            ]
        }

    async def _summarize_evidence(
        self, evidence: list[str]
    ) -> list[SourceProcedureHubSummary]:
        """Summarize ordered whole member evidence in input-fitting batches."""

        batches = pack_items(
            evidence,
            token_counts=self._summary_budget.counter.count_texts(evidence),
            token_budget=self._summary_budget.token_limit,
        )
        return [
            await self._summary_module.acall(
                request=SourceProcedureHubSummaryInput(evidence=batch)
            )
            for batch in batches
        ]

    async def _merge_summary_level(
        self, summaries: list[SourceProcedureHubSummary]
    ) -> list[SourceProcedureHubSummary]:
        """Merge one input-fitting level, carrying any trailing singleton."""

        batches = pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        )
        merged: list[SourceProcedureHubSummary] = []
        for batch in batches:
            if len(batch) == 1:
                merged.extend(batch)
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=SourceProcedureHubSummaryMergeInput(
                            summaries=batch
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Collect procedure definitions in community order."""
        return {
            'source_procedure_hub_synthesis_results_ordered': sorted(
                state.source_procedure_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SourceSemanticState) -> dict[str, object]:
        """Embed ordered procedure definitions and construct typed hubs."""
        results = state.source_procedure_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'source_procedure_hubs': [],
                'source_procedure_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.canonical_name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            SourceProcedureHub(
                source_uuid=state.source_uuid,
                canonical_name=definition.canonical_name,
                description=definition.description,
                embedding=vector,
            )
            for definition, vector in zip(definitions, vectors, strict=True)
        ]
        return {
            'source_procedure_hubs': hubs,
            'source_procedure_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _procedure_judge_input(
    index: int, candidate: SourceProcedureHubCandidate
) -> SourceProcedureHubJudgeInput:
    """Build one indexed procedure judge request."""
    return SourceProcedureHubJudgeInput(
        index=index,
        left_description=candidate.left_description,
        right_description=candidate.right_description,
    )
