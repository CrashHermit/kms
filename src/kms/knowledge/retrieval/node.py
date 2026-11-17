"""Retrieve knowledge into workflow state."""

from kms.knowledge.retrieval.search import KnowledgeSearch
from kms.knowledge.state import KnowledgeState
from kms.media.models import Material


class KnowledgeSearchNode:
    def __init__(self, search: KnowledgeSearch) -> None:
        self._search = search

    async def run(self, state: KnowledgeState) -> dict:
        context = await self._search.search(
            state['user_uuid'], Material.from_text(state['understanding'])
        )
        return {'context': context, 'retrieved_context': context}
