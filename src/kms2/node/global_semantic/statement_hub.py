"""Dispatch and collect global statement hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.config.global_semantic import GlobalStatementHubSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    GlobalStatementHub,
    GlobalStatementHubCandidate,
    GlobalStatementHubJudgeInput,
    GlobalStatementHubJudgeResult,
    GlobalStatementHubMember,
    GlobalStatementHubRerankResult,
    GlobalStatementHubSynthesisInput,
    GlobalStatementHubSynthesisMember,
    GlobalStatementHubSynthesisResult,
)
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import estimate_text_tokens
from kms2.database.global_semantic.statement_hub_repository import (
    GlobalStatementHubRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_statement_hub import (
    GlobalStatementHubModule,
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
        """Read source statement hub vector candidates."""
        candidates = (
            await self._repository.read_global_statement_hub_candidates(
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        return {'global_statement_hub_candidates': candidates}

    def dispatch_rerank(
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
            left_cost = estimate_text_tokens(left_text)
            batch: list[GlobalStatementHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_cost = estimate_text_tokens(
                    _statement_text(candidate.right_description)
                )
                if (
                    batch
                    and batch_cost + right_cost
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
                    batch_cost = left_cost
                batch.append(candidate)
                batch_cost += right_cost
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

    def dispatch_judge(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_statement_hub_judge_collect']:
        """Partition borderline statement pairs into ordered judge batches."""
        if not state.global_statement_hub_borderline_pairs:
            return 'global_statement_hub_judge_collect'
        sends: list[Send] = []
        batch: list[GlobalStatementHubCandidate] = []
        batch_cost = 0
        ordinal = 0
        for candidate in state.global_statement_hub_borderline_pairs:
            cost = estimate_text_tokens(_statement_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                sends.append(
                    Send(
                        'global_statement_hub_judge_worker',
                        {
                            'global_statement_hub_judge_ordinal': ordinal,
                            'global_statement_hub_batch': batch,
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
                    'global_statement_hub_judge_worker',
                    {
                        'global_statement_hub_judge_ordinal': ordinal,
                        'global_statement_hub_batch': batch,
                    },
                )
            )
        return sends

    async def judge_worker(
        self, state: GlobalStatementHubJudgeWorkerState
    ) -> dict[str, list[GlobalStatementHubJudgeResult]]:
        """Judge one statement borderline batch."""
        batch = state['global_statement_hub_batch']
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
        """Synthesize one statement community."""
        community = state['global_statement_hub_community']
        definition = await self._module.aforward(
            request=GlobalStatementHubSynthesisInput(
                members=[
                    GlobalStatementHubSynthesisMember(
                        description=member.description,
                    )
                    for member in community
                ]
            )
        )
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


def _statement_judge_text(candidate: GlobalStatementHubCandidate) -> str:
    """Render statement descriptions for token accounting."""
    return f'{candidate.left_description}\n{candidate.right_description}'


def _statement_judge_input(
    index: int, candidate: GlobalStatementHubCandidate
) -> GlobalStatementHubJudgeInput:
    """Build one indexed statement judge request."""
    return GlobalStatementHubJudgeInput(
        index=index,
        left_description=candidate.left_description,
        right_description=candidate.right_description,
    )


__all__ = ['GlobalStatementHubNode']
