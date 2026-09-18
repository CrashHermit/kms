"""Platform-specific ownership of local model subprocess trees."""

import os
from typing import Protocol


class OwnedProcess(Protocol):
    """Operations available to a server over its owned process tree."""

    @property
    def returncode(self) -> int | None:
        """Return the child exit code when it has exited."""
        ...

    async def wait(self) -> int:
        """Wait for the direct child to exit and return its code."""
        ...

    async def stop(self, terminate_timeout: float) -> None:
        """Terminate and reap the owned process tree idempotently."""
        ...


class ProcessOwner(Protocol):
    """Factory for owned local model processes."""

    async def spawn(self, command: list[str]) -> OwnedProcess:
        """Spawn one complete owned process tree."""
        ...


def create_process_owner() -> ProcessOwner:
    """Create the owner backend for the current operating system."""
    if os.name == 'posix':
        from kms2.local_models.posix_ownership import PosixProcessOwner

        return PosixProcessOwner()
    raise RuntimeError(
        'local-model process ownership is not implemented on this platform'
    )


__all__ = [
    'OwnedProcess',
    'ProcessOwner',
    'create_process_owner',
]
