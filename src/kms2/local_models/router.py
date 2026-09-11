"""llama.cpp router process and model-transition protocol."""

import asyncio
import time
from pathlib import Path

import httpx

from kms2.config import ModelServerProfileSettings, RouterServerSettings
from kms2.local_models.process import _ManagedLlamaServer


class _RouterServer:
    """Control the router's one configured resident model-server profile."""

    def __init__(
        self,
        settings: RouterServerSettings,
        preset_path: Path,
        api_prefix: str,
    ) -> None:
        self.settings = settings
        self.endpoint = f'http://{settings.host}:{settings.port}{api_prefix}'
        self._server = _ManagedLlamaServer(
            _router_command(settings, preset_path, api_prefix),
            self.endpoint,
            '/models',
            settings,
        )
        self._client = httpx.AsyncClient(timeout=settings.health_timeout)
        self._resident_model_server: str | None = None

    async def start(self) -> None:
        """Start the empty router process."""
        await self._server.start()

    async def ensure_model_server(self, model_server_profile: str) -> None:
        """Unload every other model-server profile, then load the requested one."""
        statuses = await self._statuses()
        for server_profile, status in statuses.items():
            if server_profile != model_server_profile and status in {
                'loaded',
                'loading',
            }:
                await self._request('/models/unload', server_profile)
                await self._wait_for(server_profile, 'unloaded')
        if statuses[model_server_profile] != 'loaded':
            await self._request('/models/load', model_server_profile)
            await self._wait_for(model_server_profile, 'loaded')
        self._resident_model_server = model_server_profile

    async def release_model_server(self) -> None:
        """Unload the router model-server profile before another GPU role begins."""
        statuses = await self._statuses()
        for model_server_profile, status in statuses.items():
            if status in {'loaded', 'loading'}:
                await self._request('/models/unload', model_server_profile)
                await self._wait_for(model_server_profile, 'unloaded')
        self._resident_model_server = None

    async def close(self) -> None:
        """Unload owned models, close HTTP resources, and stop the router."""
        try:
            if self._resident_model_server is not None:
                await self.release_model_server()
        finally:
            await self._client.aclose()
            await self._server.stop()

    async def _statuses(self) -> dict[str, str]:
        response = await self._client.get(f'{self.endpoint}/models')
        response.raise_for_status()
        return {
            item['id']: item['status']['value']
            for item in response.json()['data']
        }

    async def _request(
        self,
        path: str,
        model_server_profile: str,
    ) -> None:
        response = await self._client.post(
            f'{self.endpoint}{path}', json={'model': model_server_profile}
        )
        response.raise_for_status()

    async def _wait_for(
        self,
        model_server_profile: str,
        expected: str,
    ) -> None:
        deadline = time.monotonic() + self.settings.ready_timeout
        while time.monotonic() < deadline:
            status = (await self._statuses())[model_server_profile]
            if status == expected:
                return
            await asyncio.sleep(self.settings.poll_interval)
        raise RuntimeError(
            f'{model_server_profile} did not become {expected} within '
            f'{self.settings.ready_timeout:.0f}s'
        )


def _router_command(
    settings: RouterServerSettings,
    preset_path: Path,
    api_prefix: str,
) -> list[str]:
    """Build the strict one-model llama.cpp router command."""
    return [
        settings.executable,
        '--host',
        settings.host,
        '--port',
        str(settings.port),
        '--api-prefix',
        api_prefix,
        '--models-preset',
        str(preset_path),
        '--models-max',
        '1',
        '--no-models-autoload',
    ]


def _model_server_presets_ini(settings: RouterServerSettings) -> str:
    """Render model-server profiles as a llama.cpp models-preset document."""
    lines = ['version = 1', '']
    for model_server_profile, profile in settings.model_server_profiles.items():
        lines.extend(_model_server_preset_ini(model_server_profile, profile))
    return '\n'.join(lines)


def _model_server_preset_ini(
    model_server_profile: str,
    profile: ModelServerProfileSettings,
) -> list[str]:
    lines = [
        f'[{model_server_profile}]',
        f'model = {Path(profile.model_path).expanduser()}',
    ]
    if profile.mmproj_path is not None:
        lines.append(f'mmproj = {Path(profile.mmproj_path).expanduser()}')
    lines.append(f'ctx-size = {profile.context_size}')
    if profile.cache_type_k:
        lines.append(f'cache-type-k = {profile.cache_type_k}')
    lines.extend(
        [
            f'cache-type-v = {profile.cache_type_v}',
            f'n-gpu-layers = {profile.n_gpu_layers}',
            f'no-warmup = {str(profile.no_warmup).lower()}',
            f'parallel = {profile.parallel}',
            f'flash-attn = {profile.flash_attention}',
            f'reasoning = {profile.reasoning}',
            f'threads = {profile.threads}',
        ]
    )
    lines.append('')
    return lines
