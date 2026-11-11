"""Concurrency helpers shared across application capabilities."""

import asyncio
from collections.abc import Callable


async def run_blocking_work[T](function: Callable[..., T], *args: object) -> T:
    """Run blocking work and wait for it before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        raise
