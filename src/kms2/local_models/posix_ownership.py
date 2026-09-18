"""POSIX process-group ownership for local model servers."""

import asyncio
import os
import signal
import sys

from kms2.local_models.ownership import OwnedProcess


class PosixOwnedProcess:
    """Own one POSIX process group and its direct child."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process: asyncio.subprocess.Process | None = process

    @property
    def returncode(self) -> int | None:
        """Return the direct child exit code when available."""
        process = self._process
        return None if process is None else process.returncode

    async def wait(self) -> int:
        """Wait for the direct child to exit."""
        process = self._process
        if process is None:
            raise RuntimeError('owned process has already been reaped')
        return await process.wait()

    async def stop(self, terminate_timeout: float) -> None:
        """Terminate and reap the process group idempotently."""
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=terminate_timeout,
                )
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        else:
            await process.wait()
        self._process = None


class PosixProcessOwner:
    """Spawn local model commands in an owned POSIX process group."""

    async def spawn(self, command: list[str]) -> OwnedProcess:
        """Spawn one private process group through its platform bootstrap."""
        environment = os.environ.copy()
        if sys.platform == 'linux':
            environment['KMS2_PARENT_PID'] = str(os.getpid())
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-m',
            _posix_bootstrap_module(),
            '--',
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
        return PosixOwnedProcess(process)


def _posix_bootstrap_module() -> str:
    """Return the child bootstrap matching the current POSIX platform."""
    if sys.platform == 'linux':
        return 'kms2.local_models.linux_child'
    return 'kms2.local_models.posix_child'


__all__ = ['PosixOwnedProcess', 'PosixProcessOwner']
