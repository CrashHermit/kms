"""Owner-scoped multimodal source search."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.media.models import Material
from kms.retrieval.selection.models import (
    SearchMaterialKind,
    SearchSelectionCandidate,
)
from kms.retrieval.selection.selector import SearchSelector
from kms.source.context import project_source_block, project_source_unit
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import (
    SourceSearchHit,
    SourceUnitSearchHit,
)
from kms.source.retrieval.settings import SourceSearchSettings
from kms.source.storage.retrieval import SourceRetrievalRepository


class SourceSearchService:
    """Retrieve owned source blocks and complete source units."""

    def __init__(
        self,
        repository: SourceRetrievalRepository,
        embedder: Embedder,
        settings: SourceSearchSettings,
        *,
        selection: SearchSelector,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._settings = settings
        self._selection = selection

    async def search_sources(
        self,
        user_uuid: str,
        query: Material,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchHit]:
        """Search owned source blocks with vector similarity and selection."""
        if source_uuids == []:
            return []
        embedding = await self._query_embedding(query)
        candidates = await self._repository.search_owned_blocks(
            user_uuid,
            embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
        )
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=[
                SearchSelectionCandidate(
                    position=index,
                    material=project_source_block(candidate.block).material,
                )
                for index, candidate in enumerate(candidates)
            ],
        )
        return [
            candidates[position]
            for position in positions[: self._settings.result_limit]
        ]

    async def search_source_units(
        self,
        user_uuid: str,
        query: Material,
        *,
        source_uuids: list[str] | None = None,
        kind: SourceUnitKind | None = None,
    ) -> list[SourceUnitSearchHit]:
        """Search owned units using complete ordered native material."""
        if source_uuids == []:
            return []
        embedding = await self._query_embedding(query)
        candidates = await self._repository.search_owned_units(
            user_uuid,
            embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
            kind=kind,
        )
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_UNIT,
            candidates=[
                SearchSelectionCandidate(
                    position=index,
                    material=project_source_unit(candidate.unit),
                    unit_kind=candidate.unit.kind.value,
                )
                for index, candidate in enumerate(candidates)
            ],
        )
        return [
            candidates[position]
            for position in positions[: self._settings.result_limit]
        ]

    async def _query_embedding(self, query: Material) -> list[float]:
        """Embed the complete multimodal query once."""
        return (
            await self._embedder.embed(
                EmbeddingRequest(
                    materials=[query],
                    input_type='query',
                )
            )
        )[0]
