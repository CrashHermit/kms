"""Application services for source search and evidence reading."""

import asyncio
from pathlib import Path
from typing import cast

from kms2.config.source_search import SourceSearchSettings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_evidence import (
    SourceImage,
    SourcePassage,
    SourceProcedurePassage,
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementPassage,
    SourceStatementSearchHit,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.core.reranking import RerankerClient
from kms2.database.source.source_block_repository import SourceBlockRepository


class SourceSearchService:
    """Retrieve owned source blocks and pedagogical occurrences."""

    def __init__(
        self,
        repository: SourceBlockRepository,
        embedding_client: EmbeddingClient,
        reranker_client: RerankerClient,
        settings: SourceSearchSettings,
    ) -> None:
        self._repository = repository
        self._embedding_client = embedding_client
        self._reranker_client = reranker_client
        self._settings = settings

    async def search_sources(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchHit]:
        """Search owned source blocks and return reranker order."""
        if source_uuids == []:
            return []

        query_embedding = (await self._embedding_client.embed([query]))[0]
        candidates = await self._repository.search_owned_blocks(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
        )
        if not candidates:
            return []

        ranked = await self._reranker_client.rerank(
            query,
            [cast(str, candidate.block.content) for candidate in candidates],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        hits: list[SourceSearchHit] = []
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
        return hits

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
        query_embedding = (await self._embedding_client.embed([query]))[0]
        candidates = await self._repository.search_owned_statements(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
            is_exercise=is_exercise,
        )
        if not candidates:
            return []
        ranked = await self._reranker_client.rerank(
            query,
            [candidate.statement.content for candidate in candidates],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        return [
            SourceStatementSearchHit(
                statement=candidates[result['index']].statement,
                source=candidates[result['index']].source,
                page_indexes=candidates[result['index']].page_indexes,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]

    async def search_procedures(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceProcedureSearchHit]:
        """Search owned procedure occurrences and rerank their full content."""
        if source_uuids == []:
            return []
        query_embedding = (await self._embedding_client.embed([query]))[0]
        candidates = await self._repository.search_owned_procedures(
            user_uuid,
            query_embedding,
            candidate_limit=self._settings.candidate_limit,
            source_uuids=source_uuids,
        )
        if not candidates:
            return []
        ranked = await self._reranker_client.rerank(
            query,
            [candidate.procedure.content for candidate in candidates],
            top_n=min(self._settings.result_limit, len(candidates)),
        )
        return [
            SourceProcedureSearchHit(
                procedure=candidates[result['index']].procedure,
                source=candidates[result['index']].source,
                page_indexes=candidates[result['index']].page_indexes,
                similarity=candidates[result['index']].similarity,
                relevance_score=result['relevance_score'],
            )
            for result in ranked
        ]


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
        SourceImage(asset=asset, data=Path(asset.path).read_bytes())
        for asset in assets
    ]
