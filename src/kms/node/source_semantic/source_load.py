"""Load canonical source blocks for source-semantic processing."""

from kms.core.model.block import SourceBlock
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.langgraph.source_semantic.state import SourceSemanticState


class SourceSemanticSourceLoadNode:
    """Load the canonical source-block stream for semantic processing."""

    def __init__(self, repository: SourceBlockRepository) -> None:
        self._repository = repository

    async def run(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourceBlock]]:
        """Load persisted source blocks in canonical stream order."""
        return {'blocks': await self._repository.load_blocks(state.source_uuid)}
