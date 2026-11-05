"""User-scoped proposition embedding search and reranking."""

from kms.config.source_search import SourceSearchSettings
from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.model.knowledge.retrieval import (
    ArgumentSearchHit,
    PropositionSearchHit,
)
from kms.core.model.search import (
    SearchMaterialKind,
    SearchSelectionCandidate,
)
from kms.core.reranking import Reranker
from kms.database.knowledge.repository import KnowledgeRepository
from kms.service.search_selection import SearchSelectionService


class KnowledgeSearchService:
    """Search canonical propositions with embedding, reranking, and selection."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        embedder: Embedder,
        reranker: Reranker,
        settings: SourceSearchSettings,
        *,
        selection: SearchSelectionService,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings
        self._selection = selection

    async def search_propositions(
        self,
        user_uuid: str,
        query: str,
    ) -> list[PropositionSearchHit]:
        """Return selected user-owned propositions from the bounded shortlist."""
        query_embedding = (
            await self._embedder.embed(
                [MultimodalContent(text=query)],
                content_images=[],
                input_type='query',
            )
        )[0]
        candidates = await self._repository.search_propositions(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
        )
        if not candidates:
            return []
        ranked = await self._reranker.rerank(
            MultimodalContent(text=query),
            [
                MultimodalContent(text=candidate.proposition.content)
                for candidate in candidates
            ],
            content_images=[],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits = [
            PropositionSearchHit(
                proposition=candidates[result['index']].proposition,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
        if not hits:
            return []
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.PROPOSITION,
            candidates=[
                SearchSelectionCandidate(
                    position=position,
                    material=MultimodalContent(text=hit.proposition.content),
                )
                for position, hit in enumerate(hits)
            ],
            content_images=[],
        )
        return [hits[position] for position in positions]

    async def search_arguments(
        self,
        user_uuid: str,
        query: str,
    ) -> list[ArgumentSearchHit]:
        """Search complete user-owned arguments using their canonical text."""
        query_embedding = (
            await self._embedder.embed(
                [MultimodalContent(text=query)],
                content_images=[],
                input_type='query',
            )
        )[0]
        candidates = await self._repository.search_arguments(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
        )
        if not candidates:
            return []
        ranked = await self._reranker.rerank(
            MultimodalContent(text=query),
            [
                MultimodalContent(text=candidate.content)
                for candidate in candidates
            ],
            content_images=[],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits = [
            ArgumentSearchHit(
                record=candidates[result['index']].record,
                content=candidates[result['index']].content,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
        if not hits:
            return []
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.ARGUMENT,
            candidates=[
                SearchSelectionCandidate(
                    position=position,
                    material=MultimodalContent(text=hit.content),
                )
                for position, hit in enumerate(hits)
            ],
            content_images=[],
        )
        return [hits[position] for position in positions]
