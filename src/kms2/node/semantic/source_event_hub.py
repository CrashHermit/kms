"""Dispatch and collect source-local event hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.semantic import SourceEventHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    SourceEventHub,
    SourceEventHubCandidate,
    SourceEventHubJudgeInput,
    SourceEventHubJudgeResult,
    SourceEventHubMember,
    SourceEventHubRerankResult,
    SourceEventHubSynthesisInput,
    SourceEventHubSynthesisMember,
    SourceEventHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import estimate_text_tokens
from kms2.database.semantic.source_event_repository import SourceEventRepository
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_event_hub import SourceEventHubModule
from kms2.module.semantic.source_event_hub_judge import (
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
    ) -> None:
        self._repository = repository
        self._module = module
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings

    async def load_candidates(self, state: SemanticState) -> dict[str, object]:
        """Read event vector candidates for this source."""
        candidates = await self._repository.read_source_event_hub_candidates(
            state.source_uuid,
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'source_event_hub_candidates': candidates}

    def dispatch_rerank(
        self, state: SemanticState
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
            left_cost = estimate_text_tokens(left_text)
            batch: list[SourceEventHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_cost = estimate_text_tokens(
                    _event_text(
                        candidate.right_name, candidate.right_description
                    )
                )
                if (
                    batch
                    and batch_cost + right_cost
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
                    batch_cost = left_cost
                batch.append(candidate)
                batch_cost += right_cost
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

    def collect_rerank(self, state: SemanticState) -> dict[str, object]:
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

    def dispatch_judge(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_event_hub_judge_collect']:
        """Partition borderline event pairs into ordered judge batches."""
        if not state.source_event_hub_borderline_pairs:
            return 'source_event_hub_judge_collect'
        sends: list[Send] = []
        batch: list[SourceEventHubCandidate] = []
        batch_cost = 0
        ordinal = 0
        for candidate in state.source_event_hub_borderline_pairs:
            cost = estimate_text_tokens(_event_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                sends.append(
                    Send(
                        'source_event_hub_judge_worker',
                        {
                            'source_event_hub_judge_ordinal': ordinal,
                            'source_event_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
                batch = []
                batch_cost = 0
            batch.append(candidate)
            batch_cost += cost
        if batch:
            sends.append(
                Send(
                    'source_event_hub_judge_worker',
                    {
                        'source_event_hub_judge_ordinal': ordinal,
                        'source_event_hub_batch': batch,
                    },
                )
            )
        return sends

    async def judge_worker(
        self, state: SourceEventHubJudgeWorkerState
    ) -> dict[str, list[SourceEventHubJudgeResult]]:
        """Judge one event borderline batch."""
        batch = state['source_event_hub_batch']
        decisions = await self._judge_module.aforward(
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

    def collect_judge(self, state: SemanticState) -> dict[str, object]:
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
        self, state: SemanticState
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
        self, state: SemanticState
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
        """Synthesize one event community."""
        community = state['source_event_hub_community']
        definition = await self._module.aforward(
            request=SourceEventHubSynthesisInput(
                members=[
                    SourceEventHubSynthesisMember(
                        name=member.name, description=member.description
                    )
                    for member in community
                ]
            )
        )
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

    def collect_synthesis(self, state: SemanticState) -> dict[str, object]:
        """Collect event definitions in community order."""
        return {
            'source_event_hub_synthesis_results_ordered': sorted(
                state.source_event_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SemanticState) -> dict[str, object]:
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


def _event_judge_text(candidate: SourceEventHubCandidate) -> str:
    """Render complete event evidence for judge token accounting."""
    return (
        f'{candidate.left_name}: {candidate.left_description}\n'
        f'{candidate.right_name}: {candidate.right_description}'
    )


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


__all__ = ['SourceEventHubNode']
