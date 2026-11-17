"""Retrieve relevant owner-scoped propositions and complete arguments."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.knowledge.models import KnowledgeContext
from kms.knowledge.retrieval.settings import KnowledgeSearchSettings
from kms.knowledge.storage import KnowledgeRepository
from kms.media.models import Material
from kms.retrieval.selection.models import (
    SearchMaterialKind,
    SearchSelectionCandidate,
)
from kms.retrieval.selection.selector import SearchSelector


class KnowledgeSearch:
    def __init__(
        self,
        repository: KnowledgeRepository,
        embedder: Embedder,
        embedding_model: str,
        *,
        selection: SearchSelector,
        settings: KnowledgeSearchSettings,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._selection = selection
        self._settings = settings

    async def search(self, user_uuid: str, query: Material) -> KnowledgeContext:
        [embedding] = await self._embedder.embed(
            EmbeddingRequest(materials=[query], input_type='query')
        )
        candidates = await self._repository.search(
            user_uuid,
            embedding,
            limit=self._settings.candidate_limit,
            embedding_model=self._embedding_model,
        )
        if not candidates:
            return KnowledgeContext(matched=[], propositions=[], arguments=[])
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.PROPOSITION,
            candidates=[
                SearchSelectionCandidate(
                    position=index,
                    material=Material.from_text(proposition.statement),
                )
                for index, proposition in enumerate(candidates)
            ],
        )
        matched = [
            candidates[position]
            for position in positions[: self._settings.result_limit]
        ]
        if not matched:
            return KnowledgeContext(matched=[], propositions=[], arguments=[])
        return await self._repository.context(
            user_uuid, matched, argument_limit=self._settings.argument_limit
        )
