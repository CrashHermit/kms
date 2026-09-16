"""Filter, synthesize, and stage source-local statement hubs."""

import logging
from collections.abc import Awaitable, Callable

from kms2.config import SourceStatementHubSettings
from kms2.core.model import (
    SourceStatementHub,
    SourceStatementHubCandidate,
    SourceStatementHubJudgeInput,
    SourceStatementHubSynthesisInput,
    SourceStatementHubSynthesisMember,
)
from kms2.core.windowing import estimate_text_tokens
from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_statement_hub import SourceStatementHubModule

logger = logging.getLogger(__name__)


class SourceStatementHubNode:
    """Filter statement pairs, build communities, and stage statement hubs."""

    def __init__(
        self,
        repository: SemanticRepository,
        module: SourceStatementHubModule,
        judge_module,
        reranker,
        embedding_client,
        settings: SourceStatementHubSettings,
        schema_initializer: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._module = module
        self._judge_module = judge_module
        self._reranker = reranker
        self._embedding_client = embedding_client
        self._settings = settings
        self._schema_initializer = schema_initializer

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Filter statement pairs, detect communities, and stage hubs."""
        await self._schema_initializer()
        candidates = (
            await self._repository.read_source_statement_hub_candidates(
                state.source_uuid,
                candidate_limit=self._settings.candidate_limit,
                minimum_similarity=self._settings.minimum_similarity,
            )
        )
        accepted = await self._accepted_pairs(candidates)
        await self._repository.replace_source_statement_accepted_edges(
            state.source_uuid, accepted
        )
        communities = await self._repository.detect_source_statement_communities(
            state.source_uuid,
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        definitions = [
            await self._module.aforward(
                request=SourceStatementHubSynthesisInput(
                    members=[
                        SourceStatementHubSynthesisMember(
                            description=member.description
                        )
                        for member in community
                    ]
                )
            )
            for community in communities
        ]
        memberships = [
            [member.uuid for member in community] for community in communities
        ]
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
            'source_statement_hub_memberships': memberships,
        }

    async def _accepted_pairs(
        self, candidates: list[SourceStatementHubCandidate]
    ) -> list[SourceStatementHubCandidate]:
        """Apply reranker thresholds and statement membership judgments."""
        grouped: dict[str, list[SourceStatementHubCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)
        direct: list[SourceStatementHubCandidate] = []
        borderline: list[SourceStatementHubCandidate] = []
        scores: list[float] = []
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
                    await self._classify_reranker_batch(
                        left_text, batch, direct, borderline, scores
                    )
                    batch, batch_cost = [], left_cost
                batch.append(candidate)
                batch_cost += right_cost
            if batch:
                await self._classify_reranker_batch(
                    left_text, batch, direct, borderline, scores
                )
        judged = await self._judge_borderline(borderline)
        unique = {
            (item.left_uuid, item.right_uuid): item for item in direct + judged
        }
        logger.info(
            'source statement hub accepted pairs: %d/%d',
            len(unique),
            len(candidates),
        )
        return list(unique.values())

    async def _classify_reranker_batch(
        self,
        left_text: str,
        batch: list[SourceStatementHubCandidate],
        direct: list[SourceStatementHubCandidate],
        borderline: list[SourceStatementHubCandidate],
        scores: list[float],
    ) -> None:
        results = await self._reranker.rerank(
            left_text,
            [candidate.right_description for candidate in batch],
            top_n=None,
        )
        for result in results:
            candidate = batch[result['index']]
            score = result['relevance_score']
            scores.append(score)
            if score >= self._settings.reranker_acceptance_threshold:
                direct.append(candidate)
            elif score >= self._settings.reranker_rejection_threshold:
                borderline.append(candidate)

    async def _judge_borderline(
        self, candidates: list[SourceStatementHubCandidate]
    ) -> list[SourceStatementHubCandidate]:
        accepted: list[SourceStatementHubCandidate] = []
        batch: list[SourceStatementHubCandidate] = []
        batch_cost = 0
        for candidate in candidates:
            cost = estimate_text_tokens(
                f'{candidate.left_description}\n{candidate.right_description}'
            )
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                accepted.extend(await self._judge_batch(batch))
                batch, batch_cost = [], 0
            batch.append(candidate)
            batch_cost += cost
        if batch:
            accepted.extend(await self._judge_batch(batch))
        return accepted

    async def _judge_batch(
        self, batch: list[SourceStatementHubCandidate]
    ) -> list[SourceStatementHubCandidate]:
        decisions = await self._judge_module.aforward(
            requests=[
                SourceStatementHubJudgeInput(
                    index=index,
                    left_description=item.left_description,
                    right_description=item.right_description,
                )
                for index, item in enumerate(batch)
            ]
        )
        return [
            batch[decision.index]
            for decision in decisions
            if decision.belongs_in_same_hub
        ]


__all__ = ['SourceStatementHubNode']
