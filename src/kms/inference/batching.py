"""Pack ordered inference inputs under text and image payload budgets."""

import asyncio
from collections.abc import Callable

from kms.core.tokenization.budget import TokenBudget
from kms.inference.generation import (
    fits_generation_request,
    require_generation_capacity,
)
from kms.media.budget import ImagePayloadBudget


async def pack_content_batches[Item](
    batches: list[list[Item]],
    *,
    project: Callable[[list[Item]], dict[str, object]],
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget | None,
    is_eligible: Callable[[list[Item]], bool] | None = None,
) -> list[list[Item]]:
    """Pack complete domain requests while preserving whole-item order."""
    eligible = is_eligible or (lambda items: bool(items))

    packed: list[list[Item]] = []

    async def fits(items: list[Item]) -> bool:
        return await asyncio.to_thread(
            fits_generation_request,
            project(items),
            budget=budget,
            image_budget=image_budget,
        )

    for incoming in batches:
        current: list[Item] = []
        for item in incoming:
            candidate = [*current, item]
            if not eligible(candidate):
                current = candidate
                continue
            if await fits(candidate):
                current = candidate
                continue
            if current and eligible(current):
                packed.append(current)
            current = [item]
            if eligible(current):
                await asyncio.to_thread(
                    require_generation_capacity,
                    project(current),
                    budget=budget,
                    image_budget=image_budget,
                )
        if current and eligible(current):
            packed.append(current)
    return packed
