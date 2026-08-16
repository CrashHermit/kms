"""Lifecycle management for the local model router."""

import asyncio
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from kms import config


@dataclass
class RouterConfig:
    """Configuration for a single router-mode llama-server.

    ``start`` launches the router without a model argument so it serves
    the preset models on demand, ``endpoint`` is the base URL, and
    ``models`` maps logical stage names to router model IDs.
    """

    start: list[str]
    endpoint: str
    models: dict[str, str]
    ready_timeout: float
    poll_interval: float
    request_timeout: float
    terminate_timeout: float
    cwd: str | None = None


class RouterManager:
    """Switches models on one router-mode server via its HTTP API.

    Exactly one model is resident at a time: ``switch`` unloads every
    loaded model that is not the target, then loads the target and
    waits until it reports ``loaded``.
    """

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen | None = None

    def switch(self, name: str) -> None:
        """Makes ``name`` the sole loaded model.

        Args:
            name: A logical model key from ``config.models``.

        Raises:
            RuntimeError: If the name is unknown, the router cannot be
                started, or the model fails to load.
        """
        model_id = self._config.models.get(name)
        if model_id is None:
            raise RuntimeError(f'unknown model: {name}')
        self._ensure_router()
        statuses = self._statuses()
        if model_id not in statuses:
            raise RuntimeError(
                f'router at {self._config.endpoint} does not expose '
                f'model {model_id!r}'
            )
        for mid, status in statuses.items():
            if mid != model_id and status in ('loaded', 'loading'):
                self._unload(mid)
        if statuses.get(model_id) not in ('loaded', 'loading'):
            self._load(model_id)
        self._wait_loaded(model_id)

    async def aswitch(self, name: str) -> None:
        """Switches models without blocking the event loop."""
        await asyncio.to_thread(self.switch, name)

    def shutdown(self) -> None:
        """Stops the router server."""
        if self._proc is not None:
            _terminate(self._proc, self._config.terminate_timeout)
            self._proc = None

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


class SwitchNode:
    """A graph node that switches the active model by name."""

    def __init__(self, manager: RouterManager, name: str) -> None:
        self._manager = manager
        self._name = name

    async def run(self, state) -> dict:
        """Switches to the managed model and leaves state unchanged."""
        await self._manager.aswitch(self._name)
        return {}


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
        models=dict(serving.module_models),
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
