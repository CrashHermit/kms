"""Dispatch and collect source-local event hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.source_semantic import SourceEventHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_semantic.source_event_hub import (
    SourceEventHub,
    SourceEventHubCandidate,
    SourceEventHubJudgeInput,
    SourceEventHubJudgeResult,
    SourceEventHubMember,
    SourceEventHubRerankResult,
    SourceEventHubSummary,
    SourceEventHubSummaryInput,
    SourceEventHubSummaryMergeInput,
    SourceEventHubSummarySynthesisInput,
    SourceEventHubSynthesisInput,
    SourceEventHubSynthesisMember,
    SourceEventHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.source_semantic.source_event_repository import (
    SourceEventRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_event_hub import (
    SourceEventHubModule,
    SourceEventHubSummaryMergeModule,
    SourceEventHubSummaryModule,
)
from kms2.module.source_semantic.source_event_hub_judge import (
    SourceEventHubJudgeModule,
)


class SourceEventHubRerankWorkerState(TypedDict):
    """State supplied to one event reranker worker."""

    source_event_hub_rerank_ordinal: int
    source_event_hub_left_text: str
    source_event_hub_batch: list[SourceEventHubCandidate]


class SourceEventHubJudgeWorkerState(TypedDict):
    """State supplied to one event judge worker."""

    source_event_hub_judge_ordinal: int
    source_event_hub_batch: list[SourceEventHubCandidate]


class SourceEventHubSynthesisWorkerState(TypedDict):
    """State supplied to one event synthesis worker."""

    source_event_hub_synthesis_ordinal: int
    source_event_hub_community: list[SourceEventHubMember]


class SourceEventHubNode:
    """Run event hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: SourceEventRepository,
        module: SourceEventHubModule,
        judge_module: SourceEventHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: SourceEventHubSettings,
        *,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
        summary_module: SourceEventHubSummaryModule,
        merge_module: SourceEventHubSummaryMergeModule,
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
        """Read event vector candidates for this source."""
        candidates = await self._repository.read_source_event_hub_candidates(
            state.source_uuid,
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'source_event_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_event_hub_rerank_collect']:
        """Partition event candidates into ordered reranker batches."""
        if not state.source_event_hub_candidates:
            return 'source_event_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[SourceEventHubCandidate]] = {}
        for candidate in state.source_event_hub_candidates:
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
            batch: list[SourceEventHubCandidate] = []
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
                            'source_event_hub_rerank_worker',
                            {
                                'source_event_hub_rerank_ordinal': ordinal,
                                'source_event_hub_left_text': left_text,
                                'source_event_hub_batch': batch,
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
                        'source_event_hub_rerank_worker',
                        {
                            'source_event_hub_rerank_ordinal': ordinal,
                            'source_event_hub_left_text': left_text,
                            'source_event_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: SourceEventHubRerankWorkerState
    ) -> dict[str, list[SourceEventHubRerankResult]]:
        """Rerank one event candidate batch."""
        batch = state['source_event_hub_batch']
        results = await self._reranker.rerank(
            state['source_event_hub_left_text'],
            [
                _event_text(candidate.right_name, candidate.right_description)
                for candidate in batch
            ],
            top_n=None,
        )
        direct: list[SourceEventHubCandidate] = []
        borderline: list[SourceEventHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'source_event_hub_rerank_results': [
                SourceEventHubRerankResult(
                    ordinal=state['source_event_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: SourceSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.source_event_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'source_event_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'source_event_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    async def dispatch_judge(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_event_hub_judge_collect']:
        """Partition borderline event pairs into ordered judge batches."""
        if not state.source_event_hub_borderline_pairs:
            return 'source_event_hub_judge_collect'

        batches = pack_items(
            state.source_event_hub_borderline_pairs,
            token_counts=self._judge_budget.counter.count_texts(
                [
                    _event_judge_input(0, item).model_dump_json(
                        exclude={'index'}
                    )
                    for item in state.source_event_hub_borderline_pairs
                ]
            ),
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
        )
        return [
            Send(
                'source_event_hub_judge_worker',
                {
                    'source_event_hub_judge_ordinal': ordinal,
                    'source_event_hub_batch': batch,
                },
            )
            for ordinal, batch in enumerate(batches)
        ]

    async def judge_worker(
        self, state: SourceEventHubJudgeWorkerState
    ) -> dict[str, list[SourceEventHubJudgeResult]]:
        """Judge one event borderline batch."""
        batch = state['source_event_hub_batch']
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
            'source_event_hub_judge_results': [
                SourceEventHubJudgeResult(
                    ordinal=state['source_event_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: SourceSemanticState) -> dict[str, object]:
        """Merge direct and judged event pairs in stable order."""
        accepted = list(state.source_event_hub_direct_pairs)
        for result in sorted(
            state.source_event_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], SourceEventHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'source_event_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect event communities."""
        await self._repository.replace_source_event_accepted_edges(
            state.source_uuid, state.source_event_hub_accepted_pairs
        )
        communities = await self._repository.detect_source_event_communities(
            state.source_uuid,
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'source_event_hub_communities': communities}

    def dispatch_synthesis(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_event_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered event community."""
        if not state.source_event_hub_communities:
            return 'source_event_hub_synthesis_collect'
        sends = [
            Send(
                'source_event_hub_synthesis_worker',
                {
                    'source_event_hub_synthesis_ordinal': ordinal,
                    'source_event_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.source_event_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: SourceEventHubSynthesisWorkerState
    ) -> dict[str, list[SourceEventHubSynthesisResult]]:
        """Synthesize one event community using evidence reduction."""
        community = state['source_event_hub_community']
        original_request = SourceEventHubSynthesisInput(
            members=[
                SourceEventHubSynthesisMember(
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

            async def leaf_batch_fits(batch: list[str]) -> bool:
                return fits_token_budget(
                    token_count=self._summary_budget.counter.count_texts(
                        [
                            SourceEventHubSummaryInput(
                                evidence=batch,
                            ).model_dump_json()
                        ]
                    )[0],
                    threshold=self._summary_budget.token_limit,
                )

            current: list[SourceEventHubSummary] = []
            for batch in pack_items(
                records,
                token_counts=self._summary_budget.counter.count_texts(records),
                token_budget=self._summary_budget.token_limit,
            ):
                current.append(
                    await self._summary_module.acall(
                        request=SourceEventHubSummaryInput(
                            evidence=batch,
                        )
                    )
                )

            while True:
                reduced_request = SourceEventHubSummarySynthesisInput(
                    summaries=current
                )
                if fits_token_budget(
                    token_count=self._final_budget.counter.count_texts(
                        [reduced_request.model_dump_json()]
                    )[0],
                    threshold=self._final_budget.token_limit,
                ):
                    definition = await self._module.acall(
                        request=reduced_request
                    )
                    break
                current = await self._merge_summary_level(current)

        return {
            'source_event_hub_synthesis_results': [
                SourceEventHubSynthesisResult(
                    ordinal=state['source_event_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                    aliases=[member.name for member in community],
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[SourceEventHubSummary]
    ) -> list[SourceEventHubSummary]:
        """Merge one level while carrying a trailing singleton."""

        async def merge_batch_fits(
            batch: list[SourceEventHubSummary],
        ) -> bool:
            return fits_token_budget(
                token_count=self._merge_budget.counter.count_texts(
                    [
                        SourceEventHubSummaryMergeInput(
                            summaries=batch,
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._merge_budget.token_limit,
            )

        merged: list[SourceEventHubSummary] = []
        for batch in pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        ):
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=SourceEventHubSummaryMergeInput(
                            summaries=batch,
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Collect event definitions in community order."""
        return {
            'source_event_hub_synthesis_results_ordered': sorted(
                state.source_event_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SourceSemanticState) -> dict[str, object]:
        """Embed ordered event definitions and construct typed hubs."""
        results = state.source_event_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {'source_event_hubs': [], 'source_event_hub_memberships': []}
        vectors = await self._embedding_client.embed(
            [
                f'{definition.name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            SourceEventHub(
                source_uuid=state.source_uuid,
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
            'source_event_hubs': hubs,
            'source_event_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _event_text(name: str, description: str) -> str:
    """Render one event occurrence for reranking."""
    return f'{name}: {description}'


def _event_judge_input(
    index: int, candidate: SourceEventHubCandidate
) -> SourceEventHubJudgeInput:
    """Build one indexed event judge request."""
    return SourceEventHubJudgeInput(
        index=index,
        left_name=candidate.left_name,
        left_description=candidate.left_description,
        right_name=candidate.right_name,
        right_description=candidate.right_description,
    )
