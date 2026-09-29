"""LangGraph node for durable source-fact persistence."""

from kms2.database.source_semantic.source_fact_repository import (
    SourceFactRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceFactPersistenceNode:
    """Persist facts and their node-first evidence pointers."""

    def __init__(self, repository: SourceFactRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        """Replace the source's durable fact layer."""
        await self._repository.replace_source_facts(
            state.source_uuid,
            state.source_facts,
        )
        return {}
