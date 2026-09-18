"""LangGraph node for loading source blocks into triplet extraction."""

from kms2.core.model import SourceBlock
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.langgraph.semantic.state import SemanticState


class TripletSourceLoadNode:
    """Load the canonical source-block stream for triplet extraction."""

    def __init__(self, repository: SourceBlockRepository) -> None:
        self._repository = repository

    async def run(self, state: SemanticState) -> dict[str, list[SourceBlock]]:
        """Load persisted source blocks in canonical stream order."""
        return {'blocks': await self._repository.load_blocks(state.source_uuid)}
