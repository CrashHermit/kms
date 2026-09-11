"""GPU residency coordination for local model roles."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

type AsyncOperation = Callable[[], Awaitable[object]]
type AsyncTransition = Callable[[], Awaitable[None]]


class _GpuRole(StrEnum):
    """The mutually exclusive local model GPU roles."""

    LLM = 'llm'
    EMBEDDING = 'embedding'
    RERANKER = 'reranker'


@dataclass(frozen=True, slots=True)
class _GpuResidency:
    """One mutually exclusive GPU-resident model assignment."""

    role: _GpuRole
    model_server_profile: str | None = None


class _GpuCoordinator:
    """Serialize work while maintaining one exact GPU residency."""

    def __init__(self) -> None:
        self._work_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._closed = False
        self._resident: _GpuResidency | None = None
        self._deactivate: AsyncTransition | None = None

    async def execute(
        self,
        residency: _GpuResidency,
        activate: AsyncTransition,
        deactivate: AsyncTransition,
        operation: AsyncOperation,
    ) -> object:
        """Run one operation after making its residency assignment active."""
        async with self._state_lock:
            if self._closed:
                raise RuntimeError('local model runtime is closed')
        async with self._work_lock:
            async with self._state_lock:
                if self._closed:
                    raise RuntimeError('local model runtime is closed')
            if self._resident != residency:
                if self._deactivate is not None:
                    await self._deactivate()
                await activate()
                self._resident = residency
                self._deactivate = deactivate
            return await operation()

    async def close(self) -> None:
        """Reject new work and release the resident role after active work."""
        async with self._state_lock:
            self._closed = True
        async with self._work_lock:
            if self._deactivate is not None:
                await self._deactivate()
            self._resident = None
            self._deactivate = None
