"""Load and project source-local procedure description requests."""

from kms2.config.inference import ContextWindowSettings
from kms2.core.model import (
    SourceProcedureDescriptionInput,
    SourceProcedureDescriptionRequest,
    SourceProcedureDescriptionTarget,
)
from kms2.core.windowing import select_window
from kms2.database.semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.langgraph.semantic.state import SemanticState


class SourceProcedureDescriptionLoadNode:
    """Load procedures and attach ordered source context."""

    def __init__(
        self,
        source_repository: SourceBlockRepository,
        semantic_repository: SourceProcedureRepository,
        context_window: ContextWindowSettings,
    ) -> None:
        self._source_repository = source_repository
        self._semantic_repository = semantic_repository
        self._context_window = context_window

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Load one typed request for every persisted source procedure."""
        blocks = await self._source_repository.load_blocks(state.source_uuid)
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        procedures = await self._semantic_repository.load_source_procedures(
            state.source_uuid
        )
        ordered = []
        for procedure in procedures:
            if not all(
                uuid in positions for uuid in procedure.member_block_uuids
            ):
                continue
            ordered.append(
                (
                    min(
                        positions[uuid] for uuid in procedure.member_block_uuids
                    ),
                    procedure.uuid,
                    procedure,
                )
            )
        requests = []
        for _, _, procedure in sorted(ordered):
            target_positions = sorted(
                positions[uuid] for uuid in procedure.member_block_uuids
            )
            target = SourceProcedureDescriptionTarget.model_validate(
                procedure.model_dump()
            )
            window = select_window(
                blocks,
                target_positions,
                backward_budget=self._context_window.backward_budget,
                forward_budget=self._context_window.forward_budget,
                target_budget=self._context_window.target_budget,
            )
            requests.append(
                SourceProcedureDescriptionRequest(
                    target=target,
                    model_input=SourceProcedureDescriptionInput(
                        context_before=window.context_before,
                        target_blocks=window.target,
                        context_after=window.context_after,
                    ),
                )
            )
        return {
            'source_procedure_description_requests': requests,
            'blocks': blocks,
        }
