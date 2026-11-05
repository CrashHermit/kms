"""Build bounded extraction requests over canonical source blocks."""

from kms.config.inference import ContextWindowSettings
from kms.core.model.block import SourceBlock
from kms.core.model.source_extraction import (
    SourceExtractionContext,
    SourceExtractionRequest,
    SourceExtractionTarget,
)
from kms.core.source_context import count_source_tokens
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.core.windowing import select_window


def build_source_extraction_requests(
    blocks: list[SourceBlock],
    *,
    source_uuid: str,
    context_window: ContextWindowSettings,
    token_counters: tuple[MaterialTokenEstimator, ...],
) -> list[SourceExtractionRequest]:
    """Build contiguous token-bounded requests over the source block stream."""
    if not blocks:
        return []

    token_counts = count_source_tokens(blocks, token_counters)

    requests: list[SourceExtractionRequest] = []
    position = 0
    while position < len(blocks):
        window = select_window(
            blocks,
            [position],
            token_counts=token_counts,
            backward_budget=context_window.backward_budget,
            forward_budget=context_window.forward_budget,
            target_budget=context_window.target_budget,
        )
        target_blocks = window.target
        context_before = SourceExtractionContext(
            source_blocks=window.context_before
        )
        target_end = position + len(target_blocks)
        context_after = SourceExtractionContext(
            source_blocks=window.context_after
        )
        requests.append(
            SourceExtractionRequest(
                source_uuid=source_uuid,
                target=SourceExtractionTarget(source_blocks=target_blocks),
                context_before=context_before,
                context_after=context_after,
            )
        )
        position = target_end
    return requests
