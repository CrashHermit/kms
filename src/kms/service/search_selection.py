"""Budgeted final relevance selection for retrieval shortlists."""

import dspy

from kms.core.content import ContentImage
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.content import MultimodalContent
from kms.core.model.search import (
    SearchMaterialKind,
    SearchSelectionCandidate,
    SearchSelectionInput,
)
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.module.search.selection import SearchSelectionModule
from kms.service.content import load_content_images


class SearchSelectionService:
    """Select relevant positions from a rank-ordered, whole-candidate prefix."""

    def __init__(
        self,
        module: SearchSelectionModule,
        budget: TokenBudget,
        *,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._module = module
        self._budget = budget
        self._image_budget = image_budget

    async def select(
        self,
        *,
        query: str,
        material_kind: SearchMaterialKind,
        candidates: list[SearchSelectionCandidate],
        content_images: list[ContentImage],
    ) -> list[int]:
        """Return selected input positions after one bounded model judgment."""
        if not candidates:
            return []

        fitting: list[SearchSelectionCandidate] = []
        base_request = SearchSelectionInput(
            query=query,
            material_kind=material_kind,
            candidates=fitting,
        )
        if not self._fits(base_request, []):
            raise InputBudgetExceeded(
                'Search selection input exceeds the configured budget'
            )

        for candidate in candidates:
            prefix = [*fitting, candidate]
            request, images = _project_prefix(
                prefix, content_images, query, material_kind
            )
            if not self._fits(request, images):
                if not fitting:
                    raise InputBudgetExceeded(
                        'Search selection input exceeds the configured budget'
                    )
                break
            fitting = prefix

        request, images = _project_prefix(
            fitting, content_images, query, material_kind
        )
        payloads = await load_content_images(images, budget=self._image_budget)
        return await self._module.acall(
            request=request,
            content_images=[dspy.Image(payload) for payload in payloads],
        )

    def _fits(
        self, request: SearchSelectionInput, images: list[ContentImage]
    ) -> bool:
        counter = self._budget.counter
        if counter.estimate(text=request.model_dump_json(), images=images) > (
            self._budget.token_limit
        ):
            return False

        if len(images) > self._image_budget.max_images:
            return False
        total_bytes = 0
        total_pixels = 0
        for image in images:
            metadata = counter.image_metadata.measure(image)
            total_bytes += metadata.byte_count
            total_pixels += metadata.width * metadata.height
        return (
            total_bytes <= self._image_budget.max_bytes
            and total_pixels <= self._image_budget.max_pixels
        )


def _project_prefix(
    candidates: list[SearchSelectionCandidate],
    content_images: list[ContentImage],
    query: str,
    material_kind: SearchMaterialKind,
) -> tuple[SearchSelectionInput, list[ContentImage]]:
    """Compact a prefix's referenced image catalog and remap its positions."""
    positions: list[int] = []
    seen_positions: set[int] = set()
    for candidate in candidates:
        for position in candidate.material.image_positions:
            if position not in seen_positions:
                positions.append(position)
                seen_positions.add(position)
    remapped_positions = {
        source_position: target_position
        for target_position, source_position in enumerate(positions)
    }
    projected = [
        candidate.model_copy(
            update={
                'material': MultimodalContent(
                    text=candidate.material.text,
                    image_positions=[
                        remapped_positions[position]
                        for position in candidate.material.image_positions
                    ],
                )
            }
        )
        for candidate in candidates
    ]
    return (
        SearchSelectionInput(
            query=query,
            material_kind=material_kind,
            candidates=projected,
        ),
        [content_images[position] for position in positions],
    )
