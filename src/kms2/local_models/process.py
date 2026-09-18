"""Owned subprocess lifecycle for KMS2 local model servers."""

import asyncio
import socket
from contextlib import suppress

import httpx

from kms2.config.runtime import LlamaServerSettings
from kms2.local_models.ownership import (
    OwnedProcess,
    ProcessOwner,
)


class LocalModelError(RuntimeError):
    """Base error for local model resource failures."""


class LocalModelPortInUse(LocalModelError):
    """A configured local model port is already occupied."""


class LocalModelStartError(LocalModelError):
    """An owned local model server failed to become ready."""


class ManagedLlamaServer:
    """Own one local model server through a platform-specific owner."""

    def __init__(
        self,
        command: list[str],
        endpoint: str,
        readiness_path: str,
        settings: LlamaServerSettings,
        process_owner: ProcessOwner,
    ) -> None:
        self.command = command
        self.endpoint = endpoint.rstrip('/')
        self.readiness_path = readiness_path
        self.settings = settings
        self._process_owner = process_owner
        self._process: OwnedProcess | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the owned child is still running."""
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        """Spawn the owned server and wait for its endpoint readiness."""
        if self._process is not None:
            raise RuntimeError('local model server is already started')
        self._ensure_port_available()
        self._process = await self._process_owner.spawn(self.command)
        try:
            async with asyncio.timeout(self.settings.ready_timeout):
                await self._wait_ready()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Terminate and reap the owned process idempotently."""
        process = self._process
        if process is None:
            return
        try:
            await process.stop(self.settings.terminate_timeout)
        finally:
            self._process = None

    def _ensure_port_available(self) -> None:
        host = self.settings.host
        port = self.settings.port
        try:
            with socket.create_connection((host, port), timeout=0.1):
                raise LocalModelPortInUse(
                    f'local model port {host}:{port} is already in use'
                )
        except ConnectionRefusedError:
            return
        except TimeoutError:
            raise LocalModelPortInUse(
                f'local model port {host}:{port} is not available'
            ) from None

    async def _wait_ready(self) -> None:
        process = self._process
        assert process is not None
        exit_task = asyncio.create_task(process.wait())
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.health_timeout
            ) as client:
                while True:
                    if process.returncode is not None:
                        raise LocalModelStartError(
                            f'local model server exited with code '
                            f'{process.returncode}'
                        )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(exit_task),
                            timeout=self.settings.poll_interval,
                        )
                    except TimeoutError:
                        pass
                    else:
                        raise LocalModelStartError(
                            f'local model server exited with code '
                            f'{process.returncode}'
                        )
                    try:
                        response = await client.get(
                            f'{self.endpoint}{self.readiness_path}'
                        )
                    except (httpx.ConnectError, httpx.TimeoutException):
                        continue
                    if response.is_success:
                        return
        finally:
            if not exit_task.done():
                exit_task.cancel()
                with suppress(asyncio.CancelledError):
                    await exit_task
