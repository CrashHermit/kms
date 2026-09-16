"""Load and project source-local predicate description requests."""

from kms2.config import ContextWindowSettings
from kms2.core.model import (
    SourcePredicateDescriptionInput,
    SourcePredicateDescriptionRequest,
    SourcePredicateDescriptionTarget,
)
from kms2.core.windowing import select_window
from kms2.database.semantic.repository import SemanticRepository
from kms2.database.source.repository import SourceRepository
from kms2.langgraph.semantic.state import SemanticState


class SourcePredicateDescriptionLoadNode:
    """Load predicate occurrences and attach ordered source context."""

    def __init__(
        self,
        source_repository: SourceRepository,
        semantic_repository: SemanticRepository,
        context_window: ContextWindowSettings,
    ) -> None:
        self._source_repository = source_repository
        self._semantic_repository = semantic_repository
        self._context_window = context_window

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Load one typed request for every persisted predicate occurrence."""
        blocks = await self._source_repository.load_blocks(state.source_uuid)
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        occurrences = await self._semantic_repository.load_source_predicates(
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
            target = SourcePredicateDescriptionTarget.model_validate(
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
                SourcePredicateDescriptionRequest(
                    target=target,
                    model_input=SourcePredicateDescriptionInput(
                        term=target.predicate,
                        context_before=window.context_before,
                        target_block=window.target[0],
                        context_after=window.context_after,
                    ),
                )
            )
        return {
            'source_predicate_description_requests': requests,
            'blocks': blocks,
        }
