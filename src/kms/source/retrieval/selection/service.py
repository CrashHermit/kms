"""Budgeted final relevance selection for retrieval shortlists."""

from kms.concurrency import run_blocking_work
from kms.core.tokenization.budget import TokenBudget
from kms.inference.generation import (
    fits_generation_request,
    require_generation_capacity,
)
from kms.media.budget import ImagePayloadBudget
from kms.source.retrieval.selection.inference import SearchSelectionModule
from kms.source.retrieval.selection.models import (
    SearchMaterialKind,
    SearchSelectionCandidate,
    SearchSelectionInput,
)


class SearchSelectionService:
    """Select relevant positions from a vector-similarity-ordered prefix."""

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
    ) -> list[int]:
        """Return selected input positions after one bounded model judgment."""
        if not candidates:
            return []
        request = await run_blocking_work(
            self._request_prefix, query, material_kind, candidates
        )
        return await self._module.acall(request=request)

    def _request_prefix(
        self,
        query: str,
        material_kind: SearchMaterialKind,
        candidates: list[SearchSelectionCandidate],
    ) -> SearchSelectionInput:
        """Select a fitting prefix without running metadata I/O on the event loop."""
        fitting: list[SearchSelectionCandidate] = []
        base_request = SearchSelectionInput(
            query=query,
            material_kind=material_kind,
            candidates=fitting,
        )
        require_generation_capacity(
            {'request': base_request},
            budget=self._budget,
            image_budget=self._image_budget,
        )

        for candidate in candidates:
            prefix = [*fitting, candidate]
            request = SearchSelectionInput(
                query=query,
                material_kind=material_kind,
                candidates=prefix,
            )
            if not fits_generation_request(
                {'request': request},
                budget=self._budget,
                image_budget=self._image_budget,
            ):
                if not fitting:
                    require_generation_capacity(
                        {'request': request},
                        budget=self._budget,
                        image_budget=self._image_budget,
                    )
                break
            fitting = prefix

        return SearchSelectionInput(
            query=query,
            material_kind=material_kind,
            candidates=fitting,
        )
