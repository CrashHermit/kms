"""Owner-scoped multimodal source search."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.media.models import Material
from kms.source.context import project_source_block, project_source_material
from kms.source.retrieval.models import (
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementSearchHit,
)
from kms.source.retrieval.selection.models import (
    SearchMaterialKind,
    SearchSelectionCandidate,
)
from kms.source.retrieval.selection.service import SearchSelectionService
from kms.source.retrieval.settings import SourceSearchSettings
from kms.source.storage.retrieval import SourceRetrievalRepository


class SourceSearchService:
    """Retrieve owned source blocks and pedagogical occurrences."""

    def __init__(
        self,
        repository: SourceRetrievalRepository,
        embedder: Embedder,
        settings: SourceSearchSettings,
        *,
        selection: SearchSelectionService,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._settings = settings
        self._selection = selection

    async def search_sources(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceSearchHit]:
        """Search owned source blocks with vector similarity and final selection."""
        if source_uuids == []:
            return []

        query_embedding = (
            await self._embedder.embed(
                EmbeddingRequest(
                    materials=[Material.from_text(query)],
                    input_type='query',
                )
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

        materials = [
            project_source_block(candidate.block).material
            for candidate in candidates
        ]
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=[
                SearchSelectionCandidate(position=index, material=material)
                for index, material in enumerate(materials)
            ],
        )
        return [
            candidates[position]
            for position in positions[: self._settings.result_limit]
        ]

    async def search_statements(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
        is_exercise: bool | None = None,
    ) -> list[SourceStatementSearchHit]:
        """Search owned statement occurrences with final selection."""
        if source_uuids == []:
            return []
        query_embedding = (
            await self._embedder.embed(
                EmbeddingRequest(
                    materials=[Material.from_text(query)],
                    input_type='query',
                )
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
        passages = await self._repository.read_passages(
            user_uuid,
            list(
                dict.fromkeys(
                    block_uuid
                    for candidate in candidates
                    for block_uuid in candidate.statement.member_block_uuids
                )
            ),
            expand_memberships=True,
            before=0,
            after=0,
        )
        blocks_by_source = {
            passage.source.uuid: {
                item.block.uuid: item.block for item in passage.blocks
            }
            for passage in passages
        }
        instructions_by_statement: dict[tuple[str, str], list[str]] = {}
        for passage in passages:
            for group in passage.groups:
                if group.kind == 'instruction':
                    for statement_uuid in group.governed_statement_uuids:
                        instructions_by_statement.setdefault(
                            (passage.source.uuid, statement_uuid), []
                        ).extend(group.member_block_uuids)
        materials = []
        for candidate in candidates:
            blocks = blocks_by_source[candidate.source.uuid]
            member_ids = [
                *candidate.statement.member_block_uuids,
                *instructions_by_statement.get(
                    (candidate.source.uuid, candidate.statement.uuid), []
                ),
            ]
            materials.append(
                project_source_material(
                    candidate.statement.content,
                    [blocks[uuid] for uuid in dict.fromkeys(member_ids)],
                )
            )
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.STATEMENT,
            candidates=[
                SearchSelectionCandidate(
                    position=position,
                    material=material,
                    is_exercise=candidate.statement.is_exercise,
                )
                for position, (candidate, material) in enumerate(
                    zip(candidates, materials, strict=True)
                )
            ],
        )
        return [
            candidates[position]
            for position in positions[: self._settings.result_limit]
        ]

    async def search_procedures(
        self,
        user_uuid: str,
        query: str,
        *,
        source_uuids: list[str] | None = None,
    ) -> list[SourceProcedureSearchHit]:
        """Search owned procedures with final selection."""
        if source_uuids == []:
            return []
        query_embedding = (
            await self._embedder.embed(
                EmbeddingRequest(
                    materials=[Material.from_text(query)],
                    input_type='query',
                )
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
        passages = await self._repository.read_passages(
            user_uuid,
            list(
                dict.fromkeys(
                    block_uuid
                    for candidate in candidates
                    for block_uuid in candidate.procedure.member_block_uuids
                )
            ),
            expand_memberships=True,
            before=0,
            after=0,
        )
        blocks_by_source = {
            passage.source.uuid: {
                item.block.uuid: item.block for item in passage.blocks
            }
            for passage in passages
        }
        materials = []
        for candidate in candidates:
            blocks = blocks_by_source[candidate.source.uuid]
            materials.append(
                project_source_material(
                    candidate.procedure.content,
                    [
                        blocks[uuid]
                        for uuid in candidate.procedure.member_block_uuids
                    ],
                )
            )
        positions = await self._selection.select(
            query=query,
            material_kind=SearchMaterialKind.PROCEDURE,
            candidates=[
                SearchSelectionCandidate(position=position, material=material)
                for position, material in enumerate(materials)
            ],
        )
        return [
            candidates[position]
            for position in positions[: self._settings.result_limit]
        ]
