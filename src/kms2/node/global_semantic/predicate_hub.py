"""Dispatch and collect global predicate hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.global_semantic import GlobalPredicateHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    GlobalPredicateHub,
    GlobalPredicateHubCandidate,
    GlobalPredicateHubJudgeInput,
    GlobalPredicateHubJudgeResult,
    GlobalPredicateHubMember,
    GlobalPredicateHubRerankResult,
    GlobalPredicateHubSynthesisInput,
    GlobalPredicateHubSynthesisMember,
    GlobalPredicateHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import estimate_text_tokens
from kms2.database.global_semantic.predicate_hub_repository import (
    GlobalPredicateHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_predicate_hub import (
    GlobalPredicateHubModule,
)
from kms2.module.global_semantic.global_predicate_hub_judge import (
    GlobalPredicateHubJudgeModule,
)


class GlobalPredicateHubRerankWorkerState(TypedDict):
    """State supplied to one predicate reranker worker."""

    global_predicate_hub_rerank_ordinal: int
    global_predicate_hub_left_text: str
    global_predicate_hub_batch: list[GlobalPredicateHubCandidate]


class GlobalPredicateHubJudgeWorkerState(TypedDict):
    """State supplied to one predicate judge worker."""

    global_predicate_hub_judge_ordinal: int
    global_predicate_hub_batch: list[GlobalPredicateHubCandidate]


class GlobalPredicateHubSynthesisWorkerState(TypedDict):
    """State supplied to one predicate synthesis worker."""

    global_predicate_hub_synthesis_ordinal: int
    global_predicate_hub_community: list[GlobalPredicateHubMember]


class GlobalPredicateHubNode:
    """Run predicate hub discovery as ordered LangGraph phases."""

    def __init__(
        self,
        repository: GlobalPredicateHubRepository,
        module: GlobalPredicateHubModule,
        judge_module: GlobalPredicateHubJudgeModule,
        reranker: RerankerClient,
        embedding_client: EmbeddingClient,
        settings: GlobalPredicateHubSettings,
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
        """Read source predicate hub vector candidates."""
        candidates = (
            await self._repository.read_global_predicate_hub_candidates(
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        return {'global_predicate_hub_candidates': candidates}

    def dispatch_rerank(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_predicate_hub_rerank_collect']:
        """Partition predicate candidates into ordered reranker batches."""
        if not state.global_predicate_hub_candidates:
            return 'global_predicate_hub_rerank_collect'
        sends: list[Send] = []
        ordinal = 0
        grouped: dict[str, list[GlobalPredicateHubCandidate]] = {}
        for candidate in state.global_predicate_hub_candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        for group in grouped.values():
            left = group[0]
            left_text = _predicate_text(
                left.left_predicate,
                left.left_description,
            )
            left_cost = estimate_text_tokens(left_text)
            batch: list[GlobalPredicateHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_cost = estimate_text_tokens(
                    _predicate_text(
                        candidate.right_predicate,
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
                            'global_predicate_hub_rerank_worker',
                            {
                                'global_predicate_hub_rerank_ordinal': ordinal,
                                'global_predicate_hub_left_text': left_text,
                                'global_predicate_hub_batch': batch,
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
                        'global_predicate_hub_rerank_worker',
                        {
                            'global_predicate_hub_rerank_ordinal': ordinal,
                            'global_predicate_hub_left_text': left_text,
                            'global_predicate_hub_batch': batch,
                        },
                    )
                )
                ordinal += 1
        return sends

    async def rerank_worker(
        self, state: GlobalPredicateHubRerankWorkerState
    ) -> dict[str, list[GlobalPredicateHubRerankResult]]:
        """Rerank one predicate candidate batch."""
        batch = state['global_predicate_hub_batch']
        results = await self._reranker.rerank(
            state['global_predicate_hub_left_text'],
            [
                _predicate_text(
                    candidate.right_predicate,
                    candidate.right_description,
                )
                for candidate in batch
            ],
            top_n=None,
        )
        direct: list[GlobalPredicateHubCandidate] = []
        borderline: list[GlobalPredicateHubCandidate] = []
        for result in sorted(results, key=lambda item: item['index']):
            candidate = batch[result['index']]
            score = result['relevance_score']
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)
        return {
            'global_predicate_hub_rerank_results': [
                GlobalPredicateHubRerankResult(
                    ordinal=state['global_predicate_hub_rerank_ordinal'],
                    direct=direct,
                    borderline=borderline,
                )
            ]
        }

    def collect_rerank(self, state: GlobalSemanticState) -> dict[str, object]:
        """Collect reranker results in source order."""
        results = sorted(
            state.global_predicate_hub_rerank_results, key=lambda x: x.ordinal
        )
        return {
            'global_predicate_hub_direct_pairs': [
                candidate for result in results for candidate in result.direct
            ],
            'global_predicate_hub_borderline_pairs': [
                candidate
                for result in results
                for candidate in result.borderline
            ],
        }

    def dispatch_judge(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_predicate_hub_judge_collect']:
        """Partition borderline predicate pairs into ordered judge batches."""
        if not state.global_predicate_hub_borderline_pairs:
            return 'global_predicate_hub_judge_collect'
        sends: list[Send] = []
        batch: list[GlobalPredicateHubCandidate] = []
        batch_cost = 0
        ordinal = 0
        for candidate in state.global_predicate_hub_borderline_pairs:
            cost = estimate_text_tokens(_predicate_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                sends.append(
                    Send(
                        'global_predicate_hub_judge_worker',
                        {
                            'global_predicate_hub_judge_ordinal': ordinal,
                            'global_predicate_hub_batch': batch,
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
                    'global_predicate_hub_judge_worker',
                    {
                        'global_predicate_hub_judge_ordinal': ordinal,
                        'global_predicate_hub_batch': batch,
                    },
                )
            )
        return sends

    async def judge_worker(
        self, state: GlobalPredicateHubJudgeWorkerState
    ) -> dict[str, list[GlobalPredicateHubJudgeResult]]:
        """Judge one predicate borderline batch."""
        batch = state['global_predicate_hub_batch']
        decisions = await self._judge_module.aforward(
            requests=[
                _predicate_judge_input(index, item)
                for index, item in enumerate(batch)
            ]
        )
        accepted = [
            batch[decision.index]
            for decision in decisions
            if decision.belongs_in_same_hub
        ]
        return {
            'global_predicate_hub_judge_results': [
                GlobalPredicateHubJudgeResult(
                    ordinal=state['global_predicate_hub_judge_ordinal'],
                    accepted=accepted,
                )
            ]
        }

    def collect_judge(self, state: GlobalSemanticState) -> dict[str, object]:
        """Merge direct and judged predicate pairs in stable order."""
        accepted = list(state.global_predicate_hub_direct_pairs)
        for result in sorted(
            state.global_predicate_hub_judge_results, key=lambda x: x.ordinal
        ):
            accepted.extend(result.accepted)
        unique: dict[tuple[str, str], GlobalPredicateHubCandidate] = {}
        for candidate in accepted:
            unique.setdefault(
                (candidate.left_uuid, candidate.right_uuid), candidate
            )
        return {'global_predicate_hub_accepted_pairs': list(unique.values())}

    async def detect_communities(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Replace accepted edges and detect predicate communities."""
        await self._repository.replace_global_predicate_hub_accepted_edges(
            state.global_predicate_hub_accepted_pairs
        )
        communities = await self._repository.detect_global_predicate_hub_communities(
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        return {'global_predicate_hub_communities': communities}

    def dispatch_synthesis(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_predicate_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered predicate community."""
        if not state.global_predicate_hub_communities:
            return 'global_predicate_hub_synthesis_collect'
        sends = [
            Send(
                'global_predicate_hub_synthesis_worker',
                {
                    'global_predicate_hub_synthesis_ordinal': ordinal,
                    'global_predicate_hub_community': community,
                },
            )
            for ordinal, community in enumerate(
                state.global_predicate_hub_communities
            )
        ]
        return sends

    async def synthesis_worker(
        self, state: GlobalPredicateHubSynthesisWorkerState
    ) -> dict[str, list[GlobalPredicateHubSynthesisResult]]:
        """Synthesize one predicate community."""
        community = state['global_predicate_hub_community']
        definition = await self._module.aforward(
            request=GlobalPredicateHubSynthesisInput(
                members=[
                    GlobalPredicateHubSynthesisMember(
                        predicate=member.predicate,
                        description=member.description,
                    )
                    for member in community
                ]
            )
        )
        return {
            'global_predicate_hub_synthesis_results': [
                GlobalPredicateHubSynthesisResult(
                    ordinal=state['global_predicate_hub_synthesis_ordinal'],
                    definition=definition,
                    membership_uuids=[member.uuid for member in community],
                    aliases=[member.predicate for member in community],
                )
            ]
        }

    def collect_synthesis(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Collect predicate definitions in community order."""
        return {
            'global_predicate_hub_synthesis_results_ordered': sorted(
                state.global_predicate_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: GlobalSemanticState) -> dict[str, object]:
        """Embed ordered predicate definitions and construct typed hubs."""
        results = state.global_predicate_hub_synthesis_results_ordered
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'global_predicate_hubs': [],
                'global_predicate_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.predicate}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            GlobalPredicateHub(
                predicate=definition.predicate,
                aliases=result.aliases,
                description=definition.description,
                embedding=vector,
            )
            for result, definition, vector in zip(
                results, definitions, vectors, strict=True
            )
        ]
        return {
            'global_predicate_hubs': hubs,
            'global_predicate_hub_memberships': [
                result.membership_uuids for result in results
            ],
        }


def _predicate_text(predicate: str, description: str) -> str:
    """Render one canonical predicate for reranking."""
    return f'{predicate}: {description}'


def _predicate_judge_text(candidate: GlobalPredicateHubCandidate) -> str:
    """Render canonical predicate evidence for token accounting."""
    return (
        f'{candidate.left_predicate}: {candidate.left_description}\n'
        f'{candidate.right_predicate}: {candidate.right_description}'
    )


def _predicate_judge_input(
    index: int, candidate: GlobalPredicateHubCandidate
) -> GlobalPredicateHubJudgeInput:
    """Build one indexed predicate judge request."""
    return GlobalPredicateHubJudgeInput(
        index=index,
        left_predicate=candidate.left_predicate,
        left_description=candidate.left_description,
        right_predicate=candidate.right_predicate,
        right_description=candidate.right_description,
    )


__all__ = ['GlobalPredicateHubNode']
