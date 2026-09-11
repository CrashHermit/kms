"""Dedicated embedding and reranking llama.cpp roles."""

from pathlib import Path

from kms2.config import EmbeddingSettings, RerankerSettings
from kms2.local_models.process import _ManagedLlamaServer


class _DedicatedRole:
    """Lazily run one dedicated local retrieval model."""

    def __init__(
        self,
        settings: EmbeddingSettings | RerankerSettings,
        api_prefix: str,
        *,
        reranking: bool,
    ) -> None:
        self._server_settings = settings.server
        self.endpoint = (
            f'http://{settings.server.host}:{settings.server.port}{api_prefix}'
        )
        self._server = _ManagedLlamaServer(
            _dedicated_command(settings, api_prefix, reranking=reranking),
            self.endpoint,
            '/health',
            settings.server,
        )
        self._started = False

    async def start(self) -> None:
        """Start this role when it first receives work."""
        if not self._started:
            await self._server.start()
            self._started = True

    async def close(self) -> None:
        """Stop this role so its model is no longer GPU resident."""
        await self._server.stop()
        self._started = False


def _dedicated_command(
    settings: EmbeddingSettings | RerankerSettings,
    api_prefix: str,
    *,
    reranking: bool,
) -> list[str]:
    """Build the exact dedicated embedding or reranking command."""
    server = settings.server
    command = [
        server.executable,
        '--model',
        str(Path(settings.model.model_path).expanduser()),
        '--host',
        server.host,
        '--port',
        str(server.port),
        '--api-prefix',
        api_prefix,
        '--embedding',
        '--device',
        server.device,
    ]
    if reranking:
        command.append('--reranking')
    command.extend(
        [
            '--pooling',
            'rank' if reranking else 'last',
            '--n-gpu-layers',
            str(server.n_gpu_layers),
            '--threads',
            str(server.threads),
            '--threads-batch',
            str(server.threads_batch),
            '--ubatch-size',
            str(server.ubatch_size),
            '--ctx-size',
            str(server.context_size),
            '--parallel',
            str(server.parallel),
            '--cache-ram',
            str(server.cache_ram),
        ]
    )
    if server.no_warmup:
        command.append('--no-warmup')
    return command
