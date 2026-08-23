"""Lifecycle management for the local model router."""

import asyncio
import contextlib
import contextvars
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from kms import config


@dataclass
class RouterConfig:
    """Configuration for a single router-mode llama-server.

    ``start`` launches the router without a model argument so it serves
    preset models on demand, and ``endpoint`` is the base URL.
    """

    start: list[str]
    endpoint: str
    ready_timeout: float
    poll_interval: float
    request_timeout: float
    terminate_timeout: float
    cwd: str | None = None


class RouterManager:
    """Switches models on one router-mode server via its HTTP API.

    The manager only performs load and unload operations. Logical module
    names are resolved by configuration before callers invoke
    :meth:`ensure_model`.
    """

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._condition = asyncio.Condition()
        self._active_operations = 0
        self._switching = False
        self._resident: str | None = None

    @property
    def resident(self) -> str | None:
        """Returns the model most recently confirmed as resident."""
        return self._resident

    def ensure_model(self, model_id: str) -> None:
        """Makes ``model_id`` the sole resident model.

        Args:
            model_id: The concrete router model or preset identifier.

        Raises:
            RuntimeError: If the router cannot expose or load the model.
        """
        with self._lock:
            if self._resident == model_id:
                return
            self._ensure_model_locked(model_id)

    async def aensure_model(self, model_id: str) -> None:
        """Ensures a concrete model is resident without blocking the loop."""
        await asyncio.to_thread(self.ensure_model, model_id)

    async def aexecute(
        self,
        model_id: str,
        operation: Callable[[], Awaitable[object]],
    ) -> object:
        """Runs an operation while its model remains resident.

        Operations targeting the current model run concurrently. A different
        target waits for current operations to finish, switches once, and
        then obtains the new model lease.
        """
        async with self._condition:
            while self._switching or (
                self._active_operations and self._resident != model_id
            ):
                await self._condition.wait()
            if self._resident != model_id:
                self._switching = True
            else:
                self._active_operations += 1

        if self._switching:
            try:
                await self.aensure_model(model_id)
            except BaseException:
                async with self._condition:
                    self._switching = False
                    self._condition.notify_all()
                raise
            async with self._condition:
                self._resident = model_id
                self._switching = False
                self._active_operations += 1
                self._condition.notify_all()

        try:
            return await operation()
        finally:
            async with self._condition:
                self._active_operations -= 1
                self._condition.notify_all()

    def _ensure_model_locked(self, model_id: str) -> None:
        """Loads one concrete model while the manager lock is held."""
        self._ensure_router()
        statuses = self._statuses()
        if model_id not in statuses:
            raise RuntimeError(
                f'router at {self._config.endpoint} does not expose '
                f'model {model_id!r}'
            )
        for loaded_model, status in statuses.items():
            if loaded_model != model_id and status in ('loaded', 'loading'):
                self._unload(loaded_model)
        if statuses.get(model_id) not in ('loaded', 'loading'):
            self._load(model_id)
        self._wait_loaded(model_id)
        self._resident = model_id

    def shutdown(self) -> None:
        """Stops the router server."""
        with self._lock:
            if self._proc is not None:
                _terminate(self._proc, self._config.terminate_timeout)
                self._proc = None
            self._resident = None

    def _ensure_router(self) -> None:
        """Starts the router if its endpoint is not already answering."""
        if (
            _get_json(
                self._config.endpoint + '/models', self._config.request_timeout
            )
            is not None
        ):
            return
        self._proc = subprocess.Popen(
            self._config.start,
            cwd=self._config.cwd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + self._config.ready_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f'router exited early with code {self._proc.returncode}'
                )
            if (
                _get_json(
                    self._config.endpoint + '/models',
                    self._config.request_timeout,
                )
                is not None
            ):
                return
            time.sleep(self._config.poll_interval)
        raise RuntimeError(
            f'{self._config.endpoint} not ready within '
            f'{self._config.ready_timeout:.0f}s'
        )

    def _statuses(self) -> dict[str, str]:
        """Maps every router model ID to its current status value."""
        data = _get_json(
            self._config.endpoint + '/models', self._config.request_timeout
        )
        if data is None:
            raise RuntimeError('router /models unavailable')
        return {
            item['id']: item.get('status', {}).get('value', 'unknown')
            for item in data.get('data', [])
        }

    def _load(self, model_id: str) -> None:
        _post_json(
            self._config.endpoint + '/models/load',
            {'model': model_id},
            self._config.request_timeout,
        )

    def _unload(self, model_id: str) -> None:
        _post_json(
            self._config.endpoint + '/models/unload',
            {'model': model_id},
            self._config.request_timeout,
        )

    def _wait_loaded(self, model_id: str) -> None:
        """Polls until the model reports ``loaded``.

        Raises:
            RuntimeError: If the model reports ``failed`` or never
                becomes loaded within ``ready_timeout``.
        """
        deadline = time.monotonic() + self._config.ready_timeout
        while time.monotonic() < deadline:
            status = self._statuses().get(model_id)
            if status == 'loaded':
                return
            if status == 'failed':
                raise RuntimeError(f'{model_id} failed to load')
            time.sleep(self._config.poll_interval)
        raise RuntimeError(
            f'{model_id} not loaded within {self._config.ready_timeout:.0f}s'
        )


_CURRENT_MANAGER: contextvars.ContextVar[RouterManager | None] = (
    contextvars.ContextVar('kms_model_manager', default=None)
)


@contextlib.contextmanager
def model_manager_context(manager: RouterManager | None):
    """Binds a model manager to LLM calls in the current context."""
    token = _CURRENT_MANAGER.set(manager)
    try:
        yield
    finally:
        _CURRENT_MANAGER.reset(token)


def current_model_manager() -> RouterManager | None:
    """Returns the model manager bound to the current execution context."""
    return _CURRENT_MANAGER.get()


def default_router() -> RouterConfig:
    """Returns the default router config for local model serving.

    A single llama-server runs in router mode on ``serving.host:port``
    and swaps between the configured module models via ``/models/load`` and
    ``/models/unload``.
    """
    serving = config.get_settings().serving
    home = Path.home()
    preset_path = home / 'models' / 'kms-models.ini'
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(_preset_ini(home))
    return RouterConfig(
        start=[
            'llama-server',
            '--host',
            serving.host,
            '--port',
            str(serving.port),
            '--models-preset',
            str(preset_path),
            '--models-max',
            str(serving.max_loaded_models),
            '--no-models-autoload',
        ],
        endpoint=f'http://{serving.host}:{serving.port}',
        ready_timeout=serving.ready_timeout,
        poll_interval=serving.poll_interval,
        request_timeout=serving.request_timeout,
        terminate_timeout=serving.terminate_timeout,
    )


def _preset_ini(home: Path) -> str:
    """Returns the models-preset INI, generated from serving presets."""
    lines = ['version = 1', '']
    for name, preset in config.get_settings().serving.presets.items():
        lines.append(f'[{name}]')
        lines.append(f'model = {_expand_home(preset.model, home)}')
        if preset.mmproj:
            lines.append(f'mmproj = {_expand_home(preset.mmproj, home)}')
        if preset.ctx_size:
            lines.append(f'ctx-size = {preset.ctx_size}')
        if preset.cache_type_k:
            lines.append(f'cache-type-k = {preset.cache_type_k}')
        if preset.cache_type_v:
            lines.append(f'cache-type-v = {preset.cache_type_v}')
        if preset.n_gpu_layers:
            lines.append(f'n-gpu-layers = {preset.n_gpu_layers}')
        if preset.no_warmup:
            lines.append('no-warmup = true')
        if preset.parallel:
            lines.append(f'parallel = {preset.parallel}')
        if preset.flash_attn:
            lines.append(f'flash-attn = {preset.flash_attn}')
        if preset.reasoning:
            lines.append(f'reasoning = {preset.reasoning}')
        if preset.temperature is not None:
            lines.append(f'temp = {preset.temperature}')
        if preset.threads:
            lines.append(f'threads = {preset.threads}')
        lines.append('')
    return '\n'.join(lines)


def _expand_home(path: str, home: Path) -> str:
    """Expands a leading ``~/`` to the home directory."""
    if path.startswith('~/'):
        return str(home / path[2:])
    return path


def _get_json(url: str, timeout: float) -> dict | None:
    """Returns parsed JSON, or None if the endpoint is unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError):
        return None


def _post_json(url: str, body: dict, timeout: float) -> dict:
    """Posts JSON and returns the parsed response.

    Raises:
        RuntimeError: If the request fails or returns a non-2xx status.
    """
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f'POST {url} -> HTTP {exc.code}: {exc.read()[:200]!r}'
        ) from exc
    except OSError as exc:
        raise RuntimeError(f'POST {url} failed: {exc}') from exc


def _terminate(proc: subprocess.Popen, timeout: float) -> None:
    """Terminates a subprocess, escalating to kill on timeout."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)
