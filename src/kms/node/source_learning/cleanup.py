"""Clear generated source-learning cards before regeneration."""

from kms.database.source_learning.repository import SourceLearningRepository
from kms.langgraph.source_learning.state import SourceLearningState


class SourceLearningCleanupNode:
    """Delete generated artifacts for the requested source."""

    def __init__(self, repository: SourceLearningRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceLearningState) -> dict[str, object]:
        await self._repository.clear_source_learning(state.source_uuid)
        return {}
