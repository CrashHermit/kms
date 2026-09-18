"""llama.cpp router process and model-profile control."""

import asyncio
import tempfile
from pathlib import Path

import httpx

from kms2.config.runtime import ModelServerProfileSettings, RouterServerSettings
from kms2.local_models.ownership import ProcessOwner
from kms2.local_models.process import ManagedLlamaServer


class RouterServer:
    """Control one owned llama.cpp router process."""

    def __init__(
        self,
        settings: RouterServerSettings,
        api_prefix: str,
        process_owner: ProcessOwner,
    ) -> None:
        self.settings = settings
        self.endpoint = f'http://{settings.host}:{settings.port}{api_prefix}'
        self._preset_directory = tempfile.TemporaryDirectory(prefix='kms2-')
        preset_path = Path(self._preset_directory.name) / 'models.ini'
        preset_path.write_text(_model_server_presets_ini(settings))
        self._server = ManagedLlamaServer(
            _router_command(settings, preset_path, api_prefix),
            self.endpoint,
            '/models',
            settings,
            process_owner,
        )
        self._client = httpx.AsyncClient(timeout=settings.health_timeout)

    async def start(self) -> None:
        """Start the empty router."""
        await self._server.start()

    async def ensure_model_server(self, model_server_profile: str) -> None:
        """Unload other profiles, then load the requested profile."""
        statuses = await self._statuses()
        for server_profile, status in statuses.items():
            if server_profile != model_server_profile and status == 'loaded':
                await self._request('/models/unload', server_profile)
                await self._wait_for(server_profile, 'unloaded')
        if statuses[model_server_profile] != 'loaded':
            await self._request('/models/load', model_server_profile)
            await self._wait_for(model_server_profile, 'loaded')

    async def release_model_server(self) -> None:
        """Unload every resident model profile."""
        statuses = await self._statuses()
        for model_server_profile, status in statuses.items():
            if status == 'loaded':
                await self._request('/models/unload', model_server_profile)
                await self._wait_for(model_server_profile, 'unloaded')

    async def close(self) -> None:
        """Unload profiles, close HTTP, stop the router, and clean presets."""
        try:
            if self._server.is_running:
                await self.release_model_server()
        finally:
            await self._client.aclose()
            await self._server.stop()
            self._preset_directory.cleanup()

    async def _statuses(self) -> dict[str, str]:
        response = await self._client.get(f'{self.endpoint}/models')
        return {
            item['id']: item['status']['value']
            for item in response.json()['data']
        }

    async def _request(
        self,
        path: str,
        model_server_profile: str,
    ) -> None:
        await self._client.post(
            f'{self.endpoint}{path}', json={'model': model_server_profile}
        )

    async def _wait_for(
        self,
        model_server_profile: str,
        expected: str,
    ) -> None:
        while True:
            status = (await self._statuses())[model_server_profile]
            if status == expected:
                return
            await asyncio.sleep(self.settings.poll_interval)


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
