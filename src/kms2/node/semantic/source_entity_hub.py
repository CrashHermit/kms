"""Dispatch and collect source-local entity hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.semantic import SourceEntityHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubJudgeInput,
    SourceEntityHubJudgeResult,
    SourceEntityHubMember,
    SourceEntityHubRerankResult,
    SourceEntityHubSynthesisInput,
    SourceEntityHubSynthesisMember,
    SourceEntityHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import estimate_text_tokens
from kms2.database.semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_entity_hub import SourceEntityHubModule
from kms2.module.semantic.source_entity_hub_judge import (
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
    ) -> None:
        self._repository = repository
        self._module = module
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings

    async def load_candidates(self, state: SemanticState) -> dict[str, object]:
        """Read entity vector candidates for this source."""
        candidates = await self._repository.read_source_entity_hub_candidates(
            state.source_uuid,
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        return {'source_entity_hub_candidates': candidates}

    def dispatch_rerank(
        self, state: SemanticState
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
            left_cost = estimate_text_tokens(left_text)
            batch: list[SourceEntityHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_cost = estimate_text_tokens(
                    _entity_text(
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
                            'source_entity_hub_rerank_worker',
                            {
                                'source_entity_hub_rerank_ordinal': ordinal,
                                'source_entity_hub_left_text': left_text,
                                'source_entity_hub_batch': batch,
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

    def collect_rerank(self, state: SemanticState) -> dict[str, object]:
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

    def dispatch_judge(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_entity_hub_judge_collect']:
        """Partition borderline entity pairs into ordered judge batches."""
        if not state.source_entity_hub_borderline_pairs:
            return 'source_entity_hub_judge_collect'
        sends: list[Send] = []
        batch: list[SourceEntityHubCandidate] = []
        batch_cost = 0
        ordinal = 0
        for candidate in state.source_entity_hub_borderline_pairs:
            cost = estimate_text_tokens(_entity_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                sends.append(
                    Send(
                        'source_entity_hub_judge_worker',
                        {
                            'source_entity_hub_judge_ordinal': ordinal,
                            'source_entity_hub_batch': batch,
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
                    'source_entity_hub_judge_worker',
                    {
                        'source_entity_hub_judge_ordinal': ordinal,
                        'source_entity_hub_batch': batch,
                    },
                )
            )
        return sends

    async def judge_worker(
        self, state: SourceEntityHubJudgeWorkerState
    ) -> dict[str, list[SourceEntityHubJudgeResult]]:
        """Judge one entity borderline batch."""
        batch = state['source_entity_hub_batch']
        decisions = await self._judge_module.aforward(
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

    def collect_judge(self, state: SemanticState) -> dict[str, object]:
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
        self, state: SemanticState
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
        self, state: SemanticState
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
        """Synthesize one entity community."""
        community = state['source_entity_hub_community']
        definition = await self._module.aforward(
            request=SourceEntityHubSynthesisInput(
                members=[
                    SourceEntityHubSynthesisMember(
                        name=member.name,
                        description=member.description,
                    )
                    for member in community
                ]
            )
        )
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

    def collect_synthesis(self, state: SemanticState) -> dict[str, object]:
        """Collect entity definitions in community order."""
        return {
            'source_entity_hub_synthesis_results_ordered': sorted(
                state.source_entity_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SemanticState) -> dict[str, object]:
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


def _entity_judge_text(candidate: SourceEntityHubCandidate) -> str:
    """Render complete entity evidence for judge token accounting."""
    return (
        f'{candidate.left_name}: {candidate.left_description}\n'
        f'{candidate.right_name}: {candidate.right_description}'
    )


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


__all__ = ['SourceEntityHubNode']
