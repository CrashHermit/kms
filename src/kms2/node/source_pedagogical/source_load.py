"""Load persisted pages for source-pedagogical preparation."""

from kms2.core.model.page import SourcePage
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourcePedagogicalSourceLoadNode:
    """Load the canonical persisted pages for one pedagogical run."""

    def __init__(self, repository: SourceGraphRepository) -> None:
        self._repository = repository

    async def run(
        self, state: SourcePedagogicalState
    ) -> dict[str, list[SourcePage]]:
        """Load pages with ordered blocks, assets, and existing embeddings."""
        return {
            'source_pages': await self._repository.load_pages(state.source_uuid)
        }
