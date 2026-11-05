"""Application services for source search and evidence reading."""

import asyncio

from kms.config.source_search import SourceSearchSettings
from kms.core.content import MultimodalContentBuilder
from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.model.search import (
    SearchMaterialKind,
    SearchSelectionCandidate,
)
from kms.core.model.source_evidence import (
    SourceImage,
    SourcePassage,
    SourceProcedurePassage,
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementPassage,
    SourceStatementSearchHit,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.reranking import Reranker
from kms.core.source_context import project_source_block
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.service.search_selection import SearchSelectionService
from kms.service.source_artifacts import read_asset_bytes


class SourceSearchService:
    """Retrieve owned source blocks and pedagogical occurrences."""

    def __init__(
        self,
        repository: SourceBlockRepository,
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

    async def search_sources(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchHit]:
        """Search owned source blocks with reranking and final selection."""
        if source_uuids == []:
            return []

        query_embedding = (
            await self._embedder.embed(
                [MultimodalContent(text=query)],
                content_images=[],
                input_type='query',
            )
        )[0]
        candidates = await self._repository.search_owned_blocks(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
        )
        if not candidates:
            return []

        builder = MultimodalContentBuilder()
        query_material = builder.build(text=query)
        candidate_materials = [
            project_source_block(candidate.block, builder).material
            for candidate in candidates
        ]
        ranked = await self._reranker.rerank(
            query_material,
            candidate_materials,
            content_images=builder.images,
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits: list[SourceSearchHit] = []
        hit_materials: list[MultimodalContent] = []
        for result in ranked:
            candidate = candidates[result['index']]
            hits.append(
                SourceSearchHit(
                    source=candidate.source,
                    block=candidate.block,
                    page_indexes=candidate.page_indexes,
                    similarity=candidate.similarity,
                    relevance_score=result['relevance_score'],
                )
            )
            hit_materials.append(candidate_materials[result['index']])
        if not hits:
            return []
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=[
                SearchSelectionCandidate(position=position, material=material)
                for position, material in enumerate(hit_materials)
            ],
            content_images=builder.images,
        )
        return [hits[position] for position in positions]

    async def search_statements(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
        is_exercise: bool | None = None,
    ) -> list[SourceStatementSearchHit]:
        """Search owned statement occurrences and rerank their full content."""
        if source_uuids == []:
            return []
        query_embedding = (
            await self._embedder.embed(
                [MultimodalContent(text=query)],
                content_images=[],
                input_type='query',
            )
        )[0]
        candidates = await self._repository.search_owned_statements(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
            is_exercise=is_exercise,
        )
        if not candidates:
            return []
        ranked = await self._reranker.rerank(
            MultimodalContent(text=query),
            [
                MultimodalContent(text=candidate.statement.content)
                for candidate in candidates
            ],
            content_images=[],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits = [
            SourceStatementSearchHit(
                statement=candidates[result['index']].statement,
                source=candidates[result['index']].source,
                page_indexes=candidates[result['index']].page_indexes,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
        if not hits:
            return []
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.STATEMENT,
            candidates=[
                SearchSelectionCandidate(
                    position=position,
                    material=MultimodalContent(text=hit.statement.content),
                    is_exercise=hit.statement.is_exercise,
                )
                for position, hit in enumerate(hits)
            ],
            content_images=[],
        )
        return [hits[position] for position in positions]

    async def search_procedures(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceProcedureSearchHit]:
        """Search owned procedures with reranking and final selection."""
        if source_uuids == []:
            return []
        query_embedding = (
            await self._embedder.embed(
                [MultimodalContent(text=query)],
                content_images=[],
                input_type='query',
            )
        )[0]
        candidates = await self._repository.search_owned_procedures(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
        )
        if not candidates:
            return []
        ranked = await self._reranker.rerank(
            MultimodalContent(text=query),
            [
                MultimodalContent(text=candidate.procedure.content)
                for candidate in candidates
            ],
            content_images=[],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits = [
            SourceProcedureSearchHit(
                procedure=candidates[result['index']].procedure,
                source=candidates[result['index']].source,
                page_indexes=candidates[result['index']].page_indexes,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]
        if not hits:
            return []
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.PROCEDURE,
            candidates=[
                SearchSelectionCandidate(
                    position=position,
                    material=MultimodalContent(text=hit.procedure.content),
                )
                for position, hit in enumerate(hits)
            ],
            content_images=[],
        )
        return [hits[position] for position in positions]


class SourceEvidenceService:
    """Read complete owned passages and original source images."""

    def __init__(self, repository: SourceBlockRepository) -> None:
        self._repository = repository

    async def read_passages(
        self,
        user_uuid: str,
        block_uuids: list[str],
        *,
        expand_memberships: bool = True,
        before: int = 0,
        after: int = 0,
    ) -> list[SourcePassage]:
        """Read selected blocks, related groups, and requested neighbors."""
        return await self._repository.read_passages(
            user_uuid,
            block_uuids,
            expand_memberships=expand_memberships,
            before=before,
            after=after,
        )

    async def read_statements(
        self,
        user_uuid: str,
        statement_uuids: list[str],
    ) -> list[SourceStatementPassage]:
        """Read owned statements and their source blocks/instructions."""
        return await self._repository.read_statements(
            user_uuid,
            statement_uuids,
        )

    async def read_procedures(
        self,
        user_uuid: str,
        procedure_uuids: list[str],
    ) -> list[SourceProcedurePassage]:
        """Read owned procedures and their complete member blocks."""
        return await self._repository.read_procedures(
            user_uuid,
            procedure_uuids,
        )

    async def read_images_at_revisions(
        self,
        user_uuid: str,
        image_references: list[tuple[str, int, str]],
    ) -> list[SourceImage]:
        """Read assets only from the requested owned source revisions."""
        assets = await self._repository.load_assets_at_revisions(
            user_uuid, image_references
        )
        if len(assets) != len(image_references):
            raise ValueError(
                'Requested source image revision is no longer available'
            )
        return await asyncio.to_thread(_read_source_images, assets)

    async def read_images(
        self,
        user_uuid: str,
        asset_uuids: list[str],
    ) -> list[SourceImage]:
        """Read original image bytes after resolving authorized graph assets."""
        assets = await self._repository.load_assets(user_uuid, asset_uuids)
        if not assets:
            return []
        return await asyncio.to_thread(_read_source_images, assets)


def _read_source_images(assets: list[VisualAsset]) -> list[SourceImage]:
    """Read original asset files in the repository's request order."""
    return [
        SourceImage(asset=asset, data=read_asset_bytes(asset))
        for asset in assets
    ]
