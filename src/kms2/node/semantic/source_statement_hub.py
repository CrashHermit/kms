"""Dispatch and collect source-local statement hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.semantic import SourceStatementHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubJudgeInput,
    SourceStatementHubJudgeResult,
    SourceStatementHubMember,
    SourceStatementHubRerankResult,
    SourceStatementHubSynthesisInput,
    SourceStatementHubSynthesisMember,
    SourceStatementHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import estimate_text_tokens
from kms2.database.semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_statement_hub import SourceStatementHubModule
from kms2.module.semantic.source_statement_hub_judge import (
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
    ) -> None:
        self._repository = repository
        self._module = module
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings

    async def load_candidates(self, state: SemanticState) -> dict[str, object]:
        """Read statement vector candidates for this source."""
        candidates = (
            await self._repository.read_source_statement_hub_candidates(
                state.source_uuid,
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        return {'source_statement_hub_candidates': candidates}

    def dispatch_rerank(
        self, state: SemanticState
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
            left_cost = estimate_text_tokens(left_text)
            batch: list[SourceStatementHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_cost = estimate_text_tokens(candidate.right_description)
                if (
                    batch
                    and batch_cost + right_cost
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
                    batch_cost = left_cost
                batch.append(candidate)
                batch_cost += right_cost
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

    def collect_rerank(self, state: SemanticState) -> dict[str, object]:
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

    def dispatch_judge(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_statement_hub_judge_collect']:
        """Partition borderline statement pairs into ordered judge batches."""
        if not state.source_statement_hub_borderline_pairs:
            return 'source_statement_hub_judge_collect'
        sends: list[Send] = []
        batch: list[SourceStatementHubCandidate] = []
        batch_cost = 0
        ordinal = 0
        for candidate in state.source_statement_hub_borderline_pairs:
            cost = estimate_text_tokens(_statement_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                sends.append(
                    Send(
                        'source_statement_hub_judge_worker',
                        {
                            'source_statement_hub_judge_ordinal': ordinal,
                            'source_statement_hub_batch': batch,
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
                    'source_statement_hub_judge_worker',
                    {
                        'source_statement_hub_judge_ordinal': ordinal,
                        'source_statement_hub_batch': batch,
                    },
                )
            )
        return sends

    async def judge_worker(
        self, state: SourceStatementHubJudgeWorkerState
    ) -> dict[str, list[SourceStatementHubJudgeResult]]:
        """Judge one statement borderline batch."""
        batch = state['source_statement_hub_batch']
        decisions = await self._judge_module.aforward(
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

    def collect_judge(self, state: SemanticState) -> dict[str, object]:
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
        self, state: SemanticState
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
        self, state: SemanticState
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
        """Synthesize one statement community."""
        community = state['source_statement_hub_community']
        definition = await self._module.aforward(
            request=SourceStatementHubSynthesisInput(
                members=[
                    SourceStatementHubSynthesisMember(
                        description=member.description
                    )
                    for member in community
                ]
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

    def collect_synthesis(self, state: SemanticState) -> dict[str, object]:
        """Collect statement definitions in community order."""
        return {
            'source_statement_hub_synthesis_results_ordered': sorted(
                state.source_statement_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SemanticState) -> dict[str, object]:
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


def _statement_judge_text(candidate: SourceStatementHubCandidate) -> str:
    """Render complete statement evidence for judge token accounting."""
    return f'{candidate.left_description}\n{candidate.right_description}'


def _statement_judge_input(
    index: int, candidate: SourceStatementHubCandidate
) -> SourceStatementHubJudgeInput:
    """Build one indexed statement judge request."""
    return SourceStatementHubJudgeInput(
        index=index,
        left_description=candidate.left_description,
        right_description=candidate.right_description,
    )


__all__ = ['SourceStatementHubNode']
