"""Public lifecycle root for KMS2 local llama.cpp services."""

import tempfile
import uuid
from pathlib import Path
from types import TracebackType

import dspy
import httpx

from kms2.config import LocalModelRuntimeSettings, StageInferenceSettings
from kms2.local_models.coordinator import _GpuCoordinator
from kms2.local_models.dedicated import _DedicatedRole
from kms2.local_models.inference import (
    _EmbeddingClient,
    _predictor,
    _RerankerClient,
    _RouterProfilePredictor,
)
from kms2.local_models.router import _model_server_presets_ini, _RouterServer


class LocalModelRuntime:
    """Own KMS2's router and retrieval servers for one application run."""

    def __init__(self, settings: LocalModelRuntimeSettings) -> None:
        self._settings = settings
        self._state = 'new'
        self._coordinator = _GpuCoordinator()
        self._preset_directory: tempfile.TemporaryDirectory[str] | None = None
        self._router: _RouterServer | None = None
        self._embedding_role: _DedicatedRole | None = None
        self._reranker_role: _DedicatedRole | None = None
        self._clients: list[httpx.AsyncClient] = []
        self.embedding: _EmbeddingClient | None = None
        self.reranker: _RerankerClient | None = None

    async def start(self) -> None:
        """Start an empty router and construct lazy retrieval clients."""
        if self._state != 'new':
            raise RuntimeError('local model runtime can only be started once')
        self._state = 'starting'
        api_prefix = f'/kms2-{uuid.uuid4().hex}'
        try:
            self._preset_directory = tempfile.TemporaryDirectory(prefix='kms2-')
            preset_path = Path(self._preset_directory.name) / 'models.ini'
            preset_path.write_text(
                _model_server_presets_ini(self._settings.router)
            )
            self._router = _RouterServer(
                self._settings.router,
                preset_path,
                api_prefix,
            )
            self._embedding_role = _DedicatedRole(
                self._settings.embedding,
                api_prefix,
                reranking=False,
            )
            self._reranker_role = _DedicatedRole(
                self._settings.reranker,
                api_prefix,
                reranking=True,
            )
            embedding_client = httpx.AsyncClient(
                base_url=self._embedding_role.endpoint,
                timeout=self._settings.embedding.timeout_seconds,
            )
            reranker_client = httpx.AsyncClient(
                base_url=self._reranker_role.endpoint,
                timeout=self._settings.reranker.timeout_seconds,
            )
            self._clients = [embedding_client, reranker_client]
            self.embedding = _EmbeddingClient(
                self._settings.embedding,
                embedding_client,
                self._coordinator,
                self._embedding_role,
            )
            self.reranker = _RerankerClient(
                self._settings.reranker,
                reranker_client,
                self._coordinator,
                self._reranker_role,
            )
            await self._router.start()
        except BaseException:
            await self._dispose()
            self._state = 'failed'
            raise
        self._state = 'running'

    async def close(self) -> None:
        """Release every owned client, process, and temporary preset."""
        if self._state in {'closed', 'failed'}:
            return
        self._state = 'closing'
        await self._dispose()
        self._state = 'closed'

    async def __aenter__(self) -> 'LocalModelRuntime':
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def predictor(
        self,
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
    ) -> _RouterProfilePredictor:
        """Build a predictor for the requested model-server profile."""
        if self._state != 'running' or self._router is None:
            raise RuntimeError('local model runtime is not running')
        return _predictor(self._router, self._coordinator, inference, signature)

    async def _dispose(self) -> None:
        errors: list[BaseException] = []
        for close in (
            self._coordinator.close,
            self._close_clients,
            self._close_roles,
            self._close_router,
        ):
            try:
                await close()
            except BaseException as error:
                errors.append(error)
        if self._preset_directory is not None:
            self._preset_directory.cleanup()
            self._preset_directory = None
        self.embedding = None
        self.reranker = None
        if errors:
            raise errors[0]

    async def _close_clients(self) -> None:
        for client in self._clients:
            await client.aclose()
        self._clients = []

    async def _close_roles(self) -> None:
        for role in (self._embedding_role, self._reranker_role):
            if role is not None:
                await role.close()
        self._embedding_role = None
        self._reranker_role = None

    async def _close_router(self) -> None:
        if self._router is not None:
            await self._router.close()
        self._router = None
