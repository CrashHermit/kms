"""Load and project source-local entity description requests."""

from kms2.config.inference import ContextWindowSettings
from kms2.core.model import (
    SourceEntityDescriptionInput,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionTarget,
)
from kms2.core.windowing import select_window
from kms2.database.semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.langgraph.semantic.state import SemanticState


class SourceEntityDescriptionLoadNode:
    """Load entity occurrences and attach ordered source context."""

    def __init__(
        self,
        source_repository: SourceBlockRepository,
        semantic_repository: SourceEntityRepository,
        context_window: ContextWindowSettings,
    ) -> None:
        self._source_repository = source_repository
        self._semantic_repository = semantic_repository
        self._context_window = context_window

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Load one typed request for every persisted entity occurrence."""
        blocks = await self._source_repository.load_blocks(state.source_uuid)
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        occurrences = await self._semantic_repository.load_source_entities(
            state.source_uuid
        )
        ordered = sorted(
            occurrences,
            key=lambda occurrence: (
                positions[occurrence.source_block_uuid],
                occurrence.uuid,
            ),
        )
        requests = []
        for occurrence in ordered:
            target = SourceEntityDescriptionTarget.model_validate(
                occurrence.model_dump()
            )
            window = select_window(
                blocks,
                [positions[occurrence.source_block_uuid]],
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=self._context_window.target_budget,
            )
            requests.append(
                SourceEntityDescriptionRequest(
                    target=target,
                    model_input=SourceEntityDescriptionInput(
                        term=target.name,
                        context_before=window.context_before,
                        target_block=window.target[0],
                        context_after=window.context_after,
                    ),
                )
            )
        return {
            'source_entity_description_requests': requests,
            'blocks': blocks,
        }
