"""Source-unit search stage for knowledge preparation."""

from kms.knowledge.state import KnowledgeState
from kms.media.models import Material
from kms.source.retrieval.models import SourceUnitSearchHit
from kms.source.retrieval.search import SourceSearchService


class SourceSearchNode:
    """Find candidate source units and select them in returned order."""

    def __init__(self, search: SourceSearchService) -> None:
        self._search = search

    async def run(
        self, state: KnowledgeState
    ) -> dict[str, list[SourceUnitSearchHit] | int | bool]:
        hits = await self._search.search_source_units(
            state['user_uuid'], Material.from_text(state['understanding'])
        )
        return {
            'source_hits': hits,
            'source_index': 0,
            'has_searched_sources': True,
        }

    @staticmethod
    def select_next(state: KnowledgeState) -> dict[str, str | int]:
        """Choose the current hit and advance the workflow cursor."""
        hit = state['source_hits'][state['source_index']]
        return {
            'unit_uuid': hit.unit.uuid,
            'source_index': state['source_index'] + 1,
        }
