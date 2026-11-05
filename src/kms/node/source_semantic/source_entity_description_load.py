"""Load and project source-local entity description requests."""

from kms.config.inference import ContextWindowSettings
from kms.core.model.context import SourceBlockWindow
from kms.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionTarget,
)
from kms.core.source_context import count_source_tokens
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.windowing import select_window
from kms.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms.langgraph.source_semantic.state import SourceSemanticState


class SourceEntityDescriptionLoadNode:
    """Load entity occurrences and attach ordered source context."""

    def __init__(
        self,
        semantic_repository: SourceEntityRepository,
        context_window: ContextWindowSettings,
        *,
        token_counters: tuple[MaterialTokenEstimator, ...],
    ) -> None:
        self._semantic_repository = semantic_repository
        self._context_window = context_window
        self._token_counters = token_counters

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        """Load one typed request for every persisted entity occurrence."""
        blocks = state.blocks
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        occurrences = await self._semantic_repository.load_source_entities(
            state.source_uuid
        )
        ordered = occurrences
        token_counts = (
            count_source_tokens(blocks, self._token_counters) if ordered else []
        )
        requests = []
        for occurrence in ordered:
            target = SourceEntityDescriptionTarget.model_validate(
                occurrence.model_dump()
            )
            target_positions = [
                positions[block.uuid]
                for block in occurrence.target.source_blocks
            ]
            window = select_window(
                blocks,
                target_positions,
                token_counts=token_counts,
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=0,
            )
            requests.append(
                SourceEntityDescriptionRequest(
                    target=target,
                    window=SourceBlockWindow(
                        context_before=window.context_before,
                        target=window.target,
                        context_after=window.context_after,
                    ),
                )
            )
        return {
            'source_entity_description_requests': requests,
        }
