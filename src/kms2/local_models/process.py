"""Async process ownership for a single local llama-server."""

import asyncio
import os
import signal
from collections.abc import Sequence

import httpx

from kms2.config import LlamaServerSettings


class _ManagedLlamaServer:
    """Own one llama-server process and wait for its endpoint readiness."""

    def __init__(
        self,
        command: Sequence[str],
        endpoint: str,
        readiness_path: str,
        settings: LlamaServerSettings,
    ) -> None:
        self.command = list(command)
        self.endpoint = endpoint.rstrip('/')
        self.readiness_path = readiness_path
        self.settings = settings
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Spawn this server and wait until its owned endpoint is ready."""
        if self._process is not None:
            raise RuntimeError('llama-server is already started')
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            await self._wait_ready()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Terminate the owned process group and reap its leader."""
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(
                process.wait(), self.settings.terminate_timeout
            )
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def _wait_ready(self) -> None:
        deadline = (
            asyncio.get_running_loop().time() + self.settings.ready_timeout
        )
        async with httpx.AsyncClient(
            timeout=self.settings.health_timeout
        ) as client:
            while asyncio.get_running_loop().time() < deadline:
                process = self._process
                if process is None:
                    raise RuntimeError('llama-server process is unavailable')
                if process.returncode is not None:
                    raise RuntimeError(
                        f'llama-server exited early with code {process.returncode}'
                    )
                try:
                    response = await client.get(
                        f'{self.endpoint}{self.readiness_path}'
                    )
                except httpx.HTTPError:
                    await asyncio.sleep(self.settings.poll_interval)
                    continue
                if response.is_success:
                    return
                await asyncio.sleep(self.settings.poll_interval)
        raise RuntimeError(
            f'{self.endpoint}{self.readiness_path} was not ready within '
            f'{self.settings.ready_timeout:.0f}s'
        )
