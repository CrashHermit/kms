"""Dispatch and collect global event hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.global_semantic import GlobalEventHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    GlobalEventHub,
    GlobalEventHubCandidate,
    GlobalEventHubJudgeInput,
    GlobalEventHubJudgeResult,
    GlobalEventHubMember,
    GlobalEventHubRerankResult,
    GlobalEventHubSynthesisInput,
    GlobalEventHubSynthesisMember,
    GlobalEventHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import estimate_text_tokens
from kms2.database.global_semantic.event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_event_hub import (
    GlobalEventHubModule,
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
    ) -> None:
        self._repository = repository
        self._module = module
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings

    async def load_candidates(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Read source event hub vector candidates."""
        candidates = await self._repository.read_global_event_hub_candidates(
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'global_event_hub_candidates': candidates}

    def dispatch_rerank(
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
            left_text = _event_text(
                left.left_name,
                left.left_description,
            )
            left_cost = estimate_text_tokens(left_text)
            batch: list[GlobalEventHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_cost = estimate_text_tokens(
                    _event_text(
                        candidate.right_name,
                        candidate.right_description,
                    )
                )
                if (
                    batch
                    and batch_cost + right_cost
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
                    batch_cost = left_cost
                batch.append(candidate)
                batch_cost += right_cost
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

    def dispatch_judge(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_event_hub_judge_collect']:
        """Partition borderline event pairs into ordered judge batches."""
        if not state.global_event_hub_borderline_pairs:
            return 'global_event_hub_judge_collect'
        sends: list[Send] = []
        batch: list[GlobalEventHubCandidate] = []
        batch_cost = 0
        ordinal = 0
        for candidate in state.global_event_hub_borderline_pairs:
            cost = estimate_text_tokens(_event_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                sends.append(
                    Send(
                        'global_event_hub_judge_worker',
                        {
                            'global_event_hub_judge_ordinal': ordinal,
                            'global_event_hub_batch': batch,
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
                    'global_event_hub_judge_worker',
                    {
                        'global_event_hub_judge_ordinal': ordinal,
                        'global_event_hub_batch': batch,
                    },
                )
            )
        return sends

    async def judge_worker(
        self, state: GlobalEventHubJudgeWorkerState
    ) -> dict[str, list[GlobalEventHubJudgeResult]]:
        """Judge one event borderline batch."""
        batch = state['global_event_hub_batch']
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
        """Synthesize one event community."""
        community = state['global_event_hub_community']
        definition = await self._module.aforward(
            request=GlobalEventHubSynthesisInput(
                members=[
                    GlobalEventHubSynthesisMember(
                        name=member.name,
                        description=member.description,
                    )
                    for member in community
                ]
            )
        )
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


def _event_judge_text(candidate: GlobalEventHubCandidate) -> str:
    """Render canonical event evidence for token accounting."""
    return (
        f'{candidate.left_name}: {candidate.left_description}\n'
        f'{candidate.right_name}: {candidate.right_description}'
    )


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


__all__ = ['GlobalEventHubNode']
