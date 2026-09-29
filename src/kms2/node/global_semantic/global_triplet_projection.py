"""Project source triplet hubs into global triplet evidence."""

from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState


class GlobalTripletProjectionNode:
    """Materialize one global triplet for each source triplet hub."""

    def __init__(self, repository: GlobalTripletRepository) -> None:
        self._repository = repository

    async def run(self, state: GlobalSemanticState) -> dict[str, int]:
        """Replace the deterministic global triplet projection."""
        persisted = await self._repository.replace_global_triplets()
        return {'global_triplet_count': persisted}
