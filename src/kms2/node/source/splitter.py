"""LangGraph node for routed source-block splitting."""

from kms2.core.context_window import select_cursor_window
from kms2.core.model import (
    SourceBlock,
    SourcePage,
    SplitCandidate,
    SplitPiece,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source.splitter import (
    ExerciseSplitterModule,
    ExerciseStripRouterModule,
)


class SplitterNode:
    """Route packed source blocks and restore split output to source pages."""

    def __init__(
        self,
        router: ExerciseStripRouterModule,
        splitter: ExerciseSplitterModule,
        *,
        backward_budget: int,
        target_budget: int,
        forward_budget: int,
    ) -> None:
        self._router = router
        self._splitter = splitter
        self._backward_budget = backward_budget
        self._target_budget = target_budget
        self._forward_budget = forward_budget

    async def run(self, state: SourceState) -> dict[str, list[SourcePage]]:
        """Split routed blocks in a flat stream and restore page wrappers."""
        flat_blocks = [
            source_block
            for page in state.image_seam_pages
            for source_block in page.blocks
        ]
        replacements: dict[int, list[SplitPiece]] = {}
        cursor = 0
        while cursor < len(flat_blocks):
            cursor_at_window_start = cursor
            next_cursor, window = select_cursor_window(
                flat_blocks,
                cursor,
                backward_budget=self._backward_budget,
                target_budget=self._target_budget,
                forward_budget=self._forward_budget,
            )
            candidates: list[SplitCandidate] = []
            for target_position, target_block in enumerate(window.target):
                contains_multiple = await self._router.aforward(
                    context_before=(
                        window.context_before + window.target[:target_position]
                    ),
                    target_block=target_block,
                    context_after=(
                        window.target[target_position + 1 :]
                        + window.context_after
                    ),
                )
                if contains_multiple:
                    candidates.append(
                        SplitCandidate(
                            position=target_position,
                            source_block=target_block,
                        )
                    )

            if candidates:
                decisions = await self._splitter.aforward(
                    context_before=window.context_before,
                    candidates=candidates,
                    context_after=window.context_after,
                )
                for decision in decisions:
                    flat_position = cursor_at_window_start + decision.position
                    replacements[flat_position] = decision.pieces
            cursor = next_cursor

        split_pages: list[SourcePage] = []
        flat_position = 0
        for page in state.image_seam_pages:
            split_blocks: list[SourceBlock] = []
            for source_block in page.blocks:
                pieces = replacements.get(flat_position)
                if pieces is None:
                    split_blocks.append(source_block)
                else:
                    split_blocks.extend(
                        SourceBlock(
                            block_type=source_block.block_type,
                            content=piece.content,
                        )
                        for piece in pieces
                    )
                flat_position += 1
            split_pages.append(page.model_copy(update={'blocks': split_blocks}))

        return {'split_pages': split_pages}
