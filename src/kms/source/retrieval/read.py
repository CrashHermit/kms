"""Owner-scoped source evidence reads."""

import asyncio

from kms.source.retrieval.models import (
    SourceImage,
    SourceImageReference,
    SourcePassage,
    SourceProcedurePassage,
    SourceStatementPassage,
)
from kms.source.storage.artifacts import read_image_bytes
from kms.source.storage.retrieval import SourceRetrievalRepository


class SourceReader:
    """Read complete owned passages and original source images."""

    def __init__(self, repository: SourceRetrievalRepository) -> None:
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

    async def read_images(
        self,
        user_uuid: str,
        block_uuids: list[str],
    ) -> list[SourceImage]:
        """Read verified image bytes for the user's requested source blocks."""
        references = await self._repository.load_images(user_uuid, block_uuids)
        return await asyncio.to_thread(_read_source_images, references)


def _read_source_images(
    references: list[SourceImageReference],
) -> list[SourceImage]:
    """Read all authorized images, failing if any stored artifact is corrupt."""
    return [
        SourceImage(
            block_uuid=reference.block_uuid,
            data=read_image_bytes(reference.image_path, reference.image_sha256),
        )
        for reference in references
    ]
