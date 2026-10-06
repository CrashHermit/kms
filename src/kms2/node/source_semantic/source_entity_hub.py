"""Dispatch and collect source-local entity hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.source_semantic import SourceEntityHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubJudgeInput,
    SourceEntityHubJudgeResult,
    SourceEntityHubMember,
    SourceEntityHubRerankResult,
    SourceEntityHubSummary,
    SourceEntityHubSummaryInput,
    SourceEntityHubSummaryMergeInput,
    SourceEntityHubSummarySynthesisInput,
    SourceEntityHubSynthesisInput,
    SourceEntityHubSynthesisMember,
    SourceEntityHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import (
    TextTokenCounter,
    TokenBudget,
    fits_token_budget,
    pack_items,
)
from kms2.database.source_semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_entity_hub import (
    SourceEntityHubModule,
    SourceEntityHubSummaryMergeModule,
    SourceEntityHubSummaryModule,
)
from kms2.module.source_semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
)


class SourceEntityHubRerankWorkerState(TypedDict):
    """State supplied to one entity reranker worker."""

    source_entity_hub_rerank_ordinal: int
    source_entity_hub_left_text: str
    source_entity_hub_batch: list[SourceEntityHubCandidate]


class SourceEntityHubJudgeWorkerState(TypedDict):
    """State supplied to one entity judge worker."""

    source_entity_hub_judge_ordinal: int
    source_entity_hub_batch: list[SourceEntityHubCandidate]


class SourceEntityHubSynthesisWorkerState(TypedDict):
    """State supplied to one entity synthesis worker."""

    source_entity_hub_synthesis_ordinal: int
    source_entity_hub_community: list[SourceEntityHubMember]


class SourceEntityHubNode:
    """Run entity hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: SourceEntityRepository,
        module: SourceEntityHubModule,
        judge_module: SourceEntityHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: SourceEntityHubSettings,
        *,
        reranker_token_counter: TextTokenCounter,
        reranker_overhead_tokens: int,
        judge_budget: TokenBudget,
        summary_module: SourceEntityHubSummaryModule,
        merge_module: SourceEntityHubSummaryMergeModule,
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
        """Read entity vector candidates for this source."""
        candidates = await self._repository.read_source_entity_hub_candidates(
            state.source_uuid,
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'source_entity_hub_candidates': candidates}

    async def dispatch_rerank(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_hub_rerank_collect']:
        """Partition entity candidates into ordered reranker batches."""
        if not state.source_entity_hub_candidates:
            return 'source_entity_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[SourceEntityHubCandidate]] = {}
        for candidate in state.source_entity_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = _entity_text(left.left_name, left.left_description)
            right_texts = [
                _entity_text(candidate.right_name, candidate.right_description)
                for candidate in group
            ]
            costs = self._reranker_token_counter.count_texts(
                [left_text, *right_texts]
            )
            left_cost, *right_costs = costs
            candidate_costs = [
                left_cost + right_cost + self._reranker_overhead_tokens
                for right_cost in right_costs
            ]
            for batch in pack_items(
                group,
                token_counts=candidate_costs,
                token_budget=self._settings.reranker_token_budget,
            ):
                sends.append(
                    Send(
                        'source_entity_hub_rerank_worker',
                        {
                            'source_entity_hub_rerank_ordinal': ordinal,
                            'source_entity_hub_left_text': left_text,
                            'source_entity_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: SourceEntityHubRerankWorkerState
    ) -> dict[str, list[SourceEntityHubRerankResult]]:
        """Rerank one entity candidate batch."""
        batch = state['source_entity_hub_batch']
        results = await self._reranker.rerank(
            state['source_entity_hub_left_text'],
            [
                _entity_text(candidate.right_name, candidate.right_description)
                for candidate in batch
            ],
            top_n=None,
        )
        direct: list[SourceEntityHubCandidate] = []
        borderline: list[SourceEntityHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'source_entity_hub_rerank_results': [
                SourceEntityHubRerankResult(
                    ordinal=state['source_entity_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: SourceSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.source_entity_hub_rerank_results,
            key=lambda item: item.ordinal,
        )
        direct = [
            candidate for result in results for candidate in result.direct
        ]
        borderline = [
            candidate for result in results for candidate in result.borderline
        ]
        return {
            'source_entity_hub_direct_pairs': direct,
            'source_entity_hub_borderline_pairs': borderline,
        }

    async def dispatch_judge(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_hub_judge_collect']:
        """Partition borderline entity pairs into ordered judge batches."""
        if not state.source_entity_hub_borderline_pairs:
            return 'source_entity_hub_judge_collect'
        items = state.source_entity_hub_borderline_pairs
        requests = [
            _entity_judge_input(index, candidate)
            for index, candidate in enumerate(items)
        ]
        costs = self._judge_budget.counter.count_texts(
            [request.model_dump_json(exclude={'index'}) for request in requests]
        )
        batches = pack_items(
            items,
            token_counts=costs,
            token_budget=self._judge_budget.token_limit,
            max_items=self._settings.judge_batch_size,
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
        """Judge one entity borderline batch."""
        batch = state['source_entity_hub_batch']
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
            'source_entity_hub_judge_results': [
                SourceEntityHubJudgeResult(
                    ordinal=state['source_entity_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: SourceSemanticState) -> dict[str, object]:
        """Merge direct and judged entity pairs in stable order."""
        accepted = list(state.source_entity_hub_direct_pairs)
        for result in sorted(
            state.source_entity_hub_judge_results,
            key=lambda item: item.ordinal,
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], SourceEntityHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'source_entity_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect entity communities."""
        await self._repository.replace_source_entity_accepted_edges(
            state.source_uuid,
            state.source_entity_hub_accepted_pairs,
        )
        communities = await self._repository.detect_source_entity_communities(
            state.source_uuid,
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'source_entity_hub_communities': communities}

    def dispatch_synthesis(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_entity_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered entity community."""
        if not state.source_entity_hub_communities:
            return 'source_entity_hub_synthesis_collect'
        sends = [
            Send(
                'source_entity_hub_synthesis_worker',
                {
                    'source_entity_hub_synthesis_ordinal': ordinal,
                    'source_entity_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.source_entity_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: SourceEntityHubSynthesisWorkerState
    ) -> dict[str, list[SourceEntityHubSynthesisResult]]:
        """Synthesize one entity community using evidence reduction."""
        community = state['source_entity_hub_community']
        original_request = SourceEntityHubSynthesisInput(
            members=[
                SourceEntityHubSynthesisMember(
                    name=member.name,
                    description=member.description,
                )
                for member in community
            ]
        )
        original_cost = self._final_budget.counter.count_texts(
            [original_request.model_dump_json()]
        )[0]
        if fits_token_budget(
            token_count=original_cost,
            threshold=self._final_budget.token_limit,
        ):
            definition = await self._module.acall(request=original_request)
        else:
            records = [
                member.model_dump_json() for member in original_request.members
            ]
            record_costs = self._summary_budget.counter.count_texts(records)
            current: list[SourceEntityHubSummary] = []
            for batch in pack_items(
                records,
                token_counts=record_costs,
                token_budget=self._summary_budget.token_limit,
            ):
                current.append(
                    await self._summary_module.acall(
                        request=SourceEntityHubSummaryInput(
                            evidence=batch,
                        )
                    )
                )

            while True:
                reduced_request = SourceEntityHubSummarySynthesisInput(
                    summaries=current
                )
                reduced_cost = self._final_budget.counter.count_texts(
                    [reduced_request.model_dump_json()]
                )[0]
                if fits_token_budget(
                    token_count=reduced_cost,
                    threshold=self._final_budget.token_limit,
                ):
                    definition = await self._module.acall(
                        request=reduced_request
                    )
                    break
                current = await self._merge_summary_level(current)

        return {
            'source_entity_hub_synthesis_results': [
                SourceEntityHubSynthesisResult(
                    ordinal=state['source_entity_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                    aliases=[member.name for member in community],
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[SourceEntityHubSummary]
    ) -> list[SourceEntityHubSummary]:
        """Merge one level while carrying a trailing singleton."""
        summary_costs = self._merge_budget.counter.count_texts(
            [summary.model_dump_json() for summary in summaries]
        )
        merged: list[SourceEntityHubSummary] = []
        for batch in pack_items(
            summaries,
            token_counts=summary_costs,
            token_budget=self._merge_budget.token_limit,
        ):
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=SourceEntityHubSummaryMergeInput(
                            summaries=batch,
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Collect entity definitions in community order."""
        return {
            'source_entity_hub_synthesis_results_ordered': sorted(
                state.source_entity_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SourceSemanticState) -> dict[str, object]:
        """Embed ordered entity definitions and construct typed hubs."""
        results = state.source_entity_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'source_entity_hubs': [],
                'source_entity_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.canonical_name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            SourceEntityHub(
                source_uuid=state.source_uuid,
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
            'source_entity_hubs': hubs,
            'source_entity_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _entity_text(name: str, description: str) -> str:
    """Render one entity occurrence for reranking."""
    return f'{name}: {description}'


def _entity_judge_input(
    index: int, candidate: SourceEntityHubCandidate
) -> SourceEntityHubJudgeInput:
    """Build one indexed entity judge request."""
    return SourceEntityHubJudgeInput(
        index=index,
        left_name=candidate.left_name,
        left_description=candidate.left_description,
        right_name=candidate.right_name,
        right_description=candidate.right_description,
    )
