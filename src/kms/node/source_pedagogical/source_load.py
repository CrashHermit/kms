"""Load persisted pages for source-pedagogical preparation."""

from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourcePedagogicalSourceLoadNode:
    """Load the canonical persisted pages for one pedagogical run."""

    def __init__(self, repository: SourceGraphRepository) -> None:
        self._repository = repository

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Load canonical source pages."""
        pages = await self._repository.load_pages(state.source_uuid)
        return {'source_pages': pages}
