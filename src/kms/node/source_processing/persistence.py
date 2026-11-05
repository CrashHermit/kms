"""LangGraph node for KMS2 source persistence."""

import asyncio

from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.service.source_artifacts import SourceArtifactStore


class SourcePersistenceNode:
    """Persist finalized source media and its structural graph."""

    def __init__(
        self,
        repository: SourceGraphRepository,
        artifacts: SourceArtifactStore,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        """Finalize media off-thread, persist it, and return durable pages."""
        embedded_pages = await asyncio.to_thread(
            self._artifacts.finalize_pages,
            state.embedded_pages,
        )
        source = await self._repository.replace_source(
            state.source,
            embedded_pages,
        )
        return {'source': source, 'embedded_pages': embedded_pages}
