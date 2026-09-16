"""Filter, synthesize, and stage source-local entity hubs."""

import logging
from collections.abc import Awaitable, Callable

from kms2.config import SourceEntityHubSettings
from kms2.core.model import (
    SourceEntityHub,
    SourceEntityHubCandidate,
    SourceEntityHubJudgeInput,
    SourceEntityHubSynthesisInput,
    SourceEntityHubSynthesisMember,
)
from kms2.core.windowing import estimate_text_tokens
from kms2.database.semantic.repository import SemanticRepository
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_entity_hub import SourceEntityHubModule

logger = logging.getLogger(__name__)


class SourceEntityHubNode:
    """Filter entity pairs, build communities, and stage entity hubs."""

    def __init__(
        self,
        repository: SemanticRepository,
        module: SourceEntityHubModule,
        judge_module,
        reranker,
        embedding_client,
        settings: SourceEntityHubSettings,
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
        """Filter entity pairs, detect communities, and stage hub vertices."""
        await self._schema_initializer()
        candidates = await self._repository.read_source_entity_hub_candidates(
            state.source_uuid,
            candidate_limit=self._settings.candidate_limit,
            minimum_similarity=self._settings.minimum_similarity,
        )
        accepted = await self._accepted_pairs(candidates)
        await self._repository.replace_source_entity_accepted_edges(
            state.source_uuid,
            accepted,
        )
        communities = await self._repository.detect_source_entity_communities(
            state.source_uuid,
            max_iterations=self._settings.max_iterations,
            min_association_strength=self._settings.min_association_strength,
            minimum_community_size=self._settings.minimum_community_size,
        )
        definitions = []
        memberships = []
        aliases = []
        for community in communities:
            definitions.append(
                await self._module.aforward(
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
            )
            memberships.append([member.uuid for member in community])
            aliases.append([member.name for member in community])
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
                aliases=community_aliases,
                description=definition.description,
                embedding=vector,
            )
            for definition, community_aliases, vector in zip(
                definitions, aliases, vectors, strict=True
            )
        ]
        return {
            'source_entity_hubs': hubs,
            'source_entity_hub_memberships': memberships,
        }

    async def _accepted_pairs(
        self, candidates: list[SourceEntityHubCandidate]
    ) -> list[SourceEntityHubCandidate]:
        """Apply reranker thresholds and entity membership judgments."""
        grouped: dict[str, list[SourceEntityHubCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.left_uuid, []).append(candidate)

        scores: list[float] = []
        direct: list[SourceEntityHubCandidate] = []
        borderline: list[SourceEntityHubCandidate] = []
        for group in grouped.values():
            left = group[0]
            left_text = _entity_text(left.left_name, left.left_description)
            left_cost = estimate_text_tokens(left_text)
            batch: list[SourceEntityHubCandidate] = []
            batch_cost = left_cost
            for candidate in group:
                right_text = _entity_text(
                    candidate.right_name, candidate.right_description
                )
                right_cost = estimate_text_tokens(right_text)
                if (
                    batch
                    and batch_cost + right_cost
                    > self._settings.reranker_token_budget
                ):
                    await self._classify_reranker_batch(
                        left_text,
                        batch,
                        direct,
                        borderline,
                        scores,
                    )
                    batch = []
                    batch_cost = left_cost
                batch.append(candidate)
                batch_cost += right_cost
            if batch:
                await self._classify_reranker_batch(
                    left_text,
                    batch,
                    direct,
                    borderline,
                    scores,
                )
        judged = await self._judge_borderline(borderline)
        merged = direct + judged
        unique: dict[tuple[str, str], SourceEntityHubCandidate] = {}
        for candidate in merged:
            unique[(candidate.left_uuid, candidate.right_uuid)] = candidate
        logger.info(
            'source entity hub diagnostics: candidates=%d '
            'direct=%d borderline=%d rejected=%d judged=%d '
            'judge_accepted=%d accepted=%d score_min=%s score_median=%s '
            'score_max=%s',
            len(candidates),
            len(direct),
            len(borderline),
            len(scores) - len(direct) - len(borderline),
            len(borderline),
            len(judged),
            len(unique),
            min(scores) if scores else None,
            sorted(scores)[len(scores) // 2] if scores else None,
            max(scores) if scores else None,
        )
        return list(unique.values())

    async def _classify_reranker_batch(
        self,
        left_text: str,
        batch: list[SourceEntityHubCandidate],
        direct: list[SourceEntityHubCandidate],
        borderline: list[SourceEntityHubCandidate],
        scores: list[float],
    ) -> None:
        results = await self._reranker.rerank(
            left_text,
            [
                _entity_text(candidate.right_name, candidate.right_description)
                for candidate in batch
            ],
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
        self, candidates: list[SourceEntityHubCandidate]
    ) -> list[SourceEntityHubCandidate]:
        accepted: list[SourceEntityHubCandidate] = []
        batch: list[SourceEntityHubCandidate] = []
        batch_cost = 0
        for candidate in candidates:
            cost = estimate_text_tokens(_entity_judge_text(candidate))
            if batch and (
                len(batch) >= self._settings.judge_batch_size
                or batch_cost + cost > self._settings.judge_token_budget
            ):
                decisions = await self._judge_module.aforward(
                    requests=[
                        _entity_judge_input(index, item)
                        for index, item in enumerate(batch)
                    ]
                )
                accepted.extend(
                    batch[decision.index]
                    for decision in decisions
                    if decision.belongs_in_same_hub
                )
                batch = []
                batch_cost = 0
            batch.append(candidate)
            batch_cost += cost
        if batch:
            decisions = await self._judge_module.aforward(
                requests=[
                    _entity_judge_input(index, item)
                    for index, item in enumerate(batch)
                ]
            )
            accepted.extend(
                batch[decision.index]
                for decision in decisions
                if decision.belongs_in_same_hub
            )
        return accepted


def _entity_text(name: str, description: str) -> str:
    """Render one entity occurrence for reranking."""
    return f'{name}: {description}'


def _entity_judge_text(candidate: SourceEntityHubCandidate) -> str:
    """Render complete entity evidence for judge token accounting."""
    return (
        f'Left name: {candidate.left_name}\n'
        f'Left description: {candidate.left_description}\n'
        f'Right name: {candidate.right_name}\n'
        f'Right description: {candidate.right_description}'
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
