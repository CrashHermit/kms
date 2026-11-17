"""Owner-scoped source block and unit reads."""

import asyncio
from collections.abc import Callable
from functools import partial

from kms.concurrency import run_blocking_work
from kms.core.tokenization.budget import TokenBudget
from kms.inference.generation import (
    fits_generation_request,
    require_generation_capacity,
)
from kms.media.budget import ImagePayloadBudget
from kms.source.retrieval.models import (
    SourceBlockRead,
    SourceImage,
    SourceImageReference,
    SourceUnitMaterial,
    SourceUnitRead,
    SourceUnitWindow,
)
from kms.source.storage.artifacts import read_image_bytes
from kms.source.storage.retrieval import SourceRetrievalRepository


class SourceReader:
    """Read complete owned blocks, units, and original source images."""

    def __init__(self, repository: SourceRetrievalRepository) -> None:
        self._repository = repository

    async def read_blocks(
        self, user_uuid: str, block_uuids: list[str]
    ) -> list[SourceBlockRead]:
        """Read requested authorized blocks."""
        return await self._repository.read_blocks(user_uuid, block_uuids)

    async def read_units(
        self,
        user_uuid: str,
        unit_uuids: list[str],
        *,
        before_units: int = 0,
        after_units: int = 0,
    ) -> list[SourceUnitRead]:
        """Read requested units and bounded neighbors."""
        return await self._repository.read_units(
            user_uuid,
            unit_uuids,
            before_units=before_units,
            after_units=after_units,
        )

    async def read_unit_window(
        self,
        user_uuid: str,
        unit_uuid: str,
        *,
        before_tokens: int,
        after_tokens: int,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
        request_inputs: Callable[[SourceUnitWindow], dict[str, object]],
    ) -> SourceUnitWindow | None:
        """Read whole neighboring units that fit the caller's complete request.

        Side budgets count native image and text tokens separately from the
        target. Earlier units are considered before later units. An inaccessible
        target returns None; a target request that cannot fit raises
        InputBudgetExceeded.
        """
        reads = await self._repository.read_units(user_uuid, [unit_uuid])
        if not reads:
            return None
        target = reads[0]
        window = SourceUnitWindow(
            source=target.source,
            before=[],
            target=_unit_material(target),
            after=[],
        )
        await run_blocking_work(
            partial(
                require_generation_capacity,
                request_inputs(window),
                budget=budget,
                image_budget=image_budget,
            )
        )

        for before, side_budget in (
            (True, before_tokens),
            (False, after_tokens),
        ):
            current_uuid = unit_uuid
            side_cost = 0
            while side_budget > 0:
                neighbors = await self._repository.read_units(
                    user_uuid,
                    [current_uuid],
                    before_units=int(before),
                    after_units=int(not before),
                )
                if len(neighbors) < 2:
                    break
                neighbor = neighbors[0] if before else neighbors[-1]
                material = _unit_material(neighbor)
                cost = await run_blocking_work(
                    partial(
                        budget.counter.estimate,
                        text=material.material.text,
                        images=material.material.images,
                    )
                )
                if side_cost + cost > side_budget:
                    break
                candidate = window.model_copy(
                    update={
                        'before': [material, *window.before]
                        if before
                        else window.before,
                        'after': window.after
                        if before
                        else [*window.after, material],
                    }
                )
                fits = await run_blocking_work(
                    partial(
                        fits_generation_request,
                        request_inputs(candidate),
                        budget=budget,
                        image_budget=image_budget,
                    )
                )
                if not fits:
                    break
                window = candidate
                side_cost += cost
                current_uuid = neighbor.unit.uuid
        return window

    async def read_images(
        self,
        user_uuid: str,
        block_uuids: list[str],
    ) -> list[SourceImage]:
        """Read verified image bytes for requested owned source blocks."""
        references = await self._repository.load_images(user_uuid, block_uuids)
        return await asyncio.to_thread(_read_source_images, references)


def _unit_material(read: SourceUnitRead) -> SourceUnitMaterial:
    return SourceUnitMaterial(
        uuid=read.unit.uuid,
        kind=read.unit.kind,
        material=read.material,
    )


def _read_source_images(
    references: list[SourceImageReference],
) -> list[SourceImage]:
    """Read authorized images, failing if stored artifacts are corrupt."""
    return [
        SourceImage(
            block_uuid=reference.block_uuid,
            data=read_image_bytes(reference.image_path, reference.image_sha256),
        )
        for reference in references
    ]
