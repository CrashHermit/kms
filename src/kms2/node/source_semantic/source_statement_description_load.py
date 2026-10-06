"""Load and project source-local statement description requests."""

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.source_semantic.source_statement import (
    SourceStatementDescriptionInput,
    SourceStatementDescriptionRequest,
    SourceStatementDescriptionTarget,
)
from kms2.core.windowing import (
    TextTokenCounter,
    count_text_tokens,
    select_window,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourceStatementDescriptionLoadNode:
    """Load statements and attach ordered source context."""

    def __init__(
        self,
        source_repository: SourceBlockRepository,
        semantic_repository: SourceStatementRepository,
        context_window: ContextWindowSettings,
        *,
        token_counters: tuple[TextTokenCounter, ...],
    ) -> None:
        self._source_repository = source_repository
        self._semantic_repository = semantic_repository
        self._context_window = context_window
        self._token_counters = token_counters

    async def run(self, state: SourceSemanticState) -> dict[str, object]:
        """Load one typed request for every persisted source statement."""
        blocks = await self._source_repository.load_blocks(state.source_uuid)
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        statements = await self._semantic_repository.load_source_statements(
            state.source_uuid
        )
        ordered = []
        for statement in statements:
            if not all(
                uuid in positions for uuid in statement.member_block_uuids
            ):
                continue
            ordered.append(
                (
                    min(
                        positions[uuid] for uuid in statement.member_block_uuids
                    ),
                    statement.uuid,
                    statement,
                )
            )
        token_counts = (
            count_text_tokens(
                [block.content for block in blocks], self._token_counters
            )
            if ordered
            else []
        )
        requests = []
        for _, _, statement in sorted(ordered):
            target_positions = sorted(
                positions[uuid] for uuid in statement.member_block_uuids
            )
            target = SourceStatementDescriptionTarget.model_validate(
                statement.model_dump()
            )
            window = select_window(
                blocks,
                target_positions,
                token_counts=token_counts,
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=self._context_window.target_budget,
            )
            requests.append(
                SourceStatementDescriptionRequest(
                    target=target,
                    model_input=SourceStatementDescriptionInput(
                        context_before=window.context_before,
                        target_blocks=window.target,
                        context_after=window.context_after,
                    ),
                )
            )
        return {
            'source_statement_description_requests': requests,
            'blocks': blocks,
        }
