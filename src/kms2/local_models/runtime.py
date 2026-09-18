"""Public lifecycle root for KMS2 local model services."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import TypedDict, cast

import dspy
import httpx

from kms2.config.inference import PredictorStrategy, StageInferenceSettings
from kms2.config.runtime import (
    DedicatedLlamaServerSettings,
    EmbeddingSettings,
    LocalModelRuntimeSettings,
    RerankerSettings,
)
from kms2.core.reranking import RerankResult
from kms2.local_models.ownership import (
    ProcessOwner,
    create_process_owner,
)
from kms2.local_models.process import ManagedLlamaServer
from kms2.local_models.router import RouterServer


class _EmbeddingResponseItem(TypedDict):
    embedding: list[float]


class _EmbeddingResponse(TypedDict):
    data: list[_EmbeddingResponseItem]


class _RerankResponse(TypedDict):
    results: list[RerankResult]


class ResidentRole(StrEnum):
    """One mutually exclusive GPU-resident local model role."""

    LLM = 'llm'
    EMBEDDING = 'embedding'
    RERANKER = 'reranker'


Operation = Callable[[], Awaitable[object]]


def _dedicated_server_command(
    model_path: str,
    settings: DedicatedLlamaServerSettings,
    api_prefix: str,
) -> list[str]:
    """Build shared llama-server options for one dedicated role."""
    return [
        settings.executable,
        '--model',
        str(Path(model_path).expanduser()),
        '--host',
        settings.host,
        '--port',
        str(settings.port),
        '--api-prefix',
        api_prefix,
        '--embedding',
        '--device',
        settings.device,
        '--n-gpu-layers',
        str(settings.n_gpu_layers),
        '--threads',
        str(settings.threads),
        '--threads-batch',
        str(settings.threads_batch),
        '--ubatch-size',
        str(settings.ubatch_size),
        '--ctx-size',
        str(settings.context_size),
        '--parallel',
        str(settings.parallel),
        '--cache-ram',
        str(settings.cache_ram),
    ] + (['--no-warmup'] if settings.no_warmup else [])


def _embedding_command(
    settings: EmbeddingSettings,
    api_prefix: str,
) -> list[str]:
    """Build the embedding server command."""
    return [
        *_dedicated_server_command(
            settings.model.model_path,
            settings.server,
            api_prefix,
        ),
        '--pooling',
        'last',
    ]


def _reranker_command(
    settings: RerankerSettings,
    api_prefix: str,
) -> list[str]:
    """Build the reranker server command."""
    return [
        *_dedicated_server_command(
            settings.model.model_path,
            settings.server,
            api_prefix,
        ),
        '--reranking',
        '--pooling',
        'rank',
    ]


class LocalModelRuntime:
    """Own KMS2 local model processes and GPU role transitions."""

    def __init__(self, settings: LocalModelRuntimeSettings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._state = 'stopped'
        self._active_role: tuple[ResidentRole, str | None] | None = None
        self._process_owner: ProcessOwner = create_process_owner()
        self._router: RouterServer | None = None
        self._embedding_server: ManagedLlamaServer | None = None
        self._reranker_server: ManagedLlamaServer | None = None
        self._embedding_client: httpx.AsyncClient | None = None
        self._reranker_client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start the owned router and prepare lazy retrieval resources."""
        async with self._lock:
            if self._state != 'stopped':
                raise RuntimeError('local model runtime is already started')
            self._state = 'starting'
            api_prefix = f'/kms2-{uuid.uuid4().hex}'
            self._router = RouterServer(
                self._settings.router,
                api_prefix,
                self._process_owner,
            )
            self._embedding_server = ManagedLlamaServer(
                _embedding_command(self._settings.embedding, api_prefix),
                f'http://{self._settings.embedding.server.host}:'
                f'{self._settings.embedding.server.port}{api_prefix}',
                '/health',
                self._settings.embedding.server,
                self._process_owner,
            )
            self._reranker_server = ManagedLlamaServer(
                _reranker_command(self._settings.reranker, api_prefix),
                f'http://{self._settings.reranker.server.host}:'
                f'{self._settings.reranker.server.port}{api_prefix}',
                '/health',
                self._settings.reranker.server,
                self._process_owner,
            )
            self._embedding_client = httpx.AsyncClient(
                base_url=self._embedding_server.endpoint,
                timeout=self._settings.embedding.timeout_seconds,
            )
            self._reranker_client = httpx.AsyncClient(
                base_url=self._reranker_server.endpoint,
                timeout=self._settings.reranker.timeout_seconds,
            )
            try:
                await self._router.start()
            except BaseException:
                try:
                    await self._close_resources()
                finally:
                    self._state = 'stopped'
                raise
            self._state = 'started'

    async def close(self) -> None:
        """Release every owned local-model resource idempotently."""
        async with self._lock:
            if self._state == 'stopped':
                return
            self._state = 'closing'
            self._active_role = None
            try:
                await self._close_resources()
            finally:
                self._state = 'stopped'

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ordered text inputs through the resident embedding server."""

        async def operation() -> object:
            client = self._embedding_client
            assert client is not None
            vectors: list[list[float]] = []
            batch_size = self._settings.embedding.batch_size
            for start in range(0, len(texts), batch_size):
                response = await client.post(
                    '/v1/embeddings',
                    json={
                        'model': self._settings.embedding.model.model_id,
                        'input': texts[start : start + batch_size],
                    },
                )
                payload = cast(_EmbeddingResponse, response.json())
                vectors.extend(item['embedding'] for item in payload['data'])
            return vectors

        return cast(
            list[list[float]],
            await self._execute(ResidentRole.EMBEDDING, None, operation),
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents through the resident reranker server."""

        async def operation() -> object:
            client = self._reranker_client
            assert client is not None
            body: dict[str, object] = {
                'model': self._settings.reranker.model.model_id,
                'query': query,
                'documents': documents,
            }
            if top_n is not None:
                body['top_n'] = top_n
            response = await client.post('/v1/rerank', json=body)
            payload = cast(_RerankResponse, response.json())
            return payload['results']

        return cast(
            list[RerankResult],
            await self._execute(ResidentRole.RERANKER, None, operation),
        )

    def predictor(
        self,
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
    ) -> 'RuntimePredictor':
        """Build a DSPy predictor controlled by this runtime."""
        router = self._router
        if router is None:
            raise RuntimeError('local model runtime has not started')
        lm = dspy.LM(
            inference.model_server_profile,
            api_base=f'{router.endpoint}/v1',
            api_key='not-needed',
            custom_llm_provider='openai',
            temperature=inference.temperature,
            max_tokens=inference.max_tokens,
            num_retries=inference.num_retries,
            cache=True,
        )
        predictor: dspy.Module
        if inference.strategy is PredictorStrategy.PREDICT:
            predictor = dspy.Predict(signature)
        else:
            predictor = dspy.ChainOfThought(signature)
        predictor.set_lm(lm)
        return RuntimePredictor(self, inference.model_server_profile, predictor)

    async def _execute(
        self,
        role: ResidentRole,
        model_server_profile: str | None,
        operation: Operation,
    ) -> object:
        async with self._lock:
            if self._state != 'started':
                raise RuntimeError('local model runtime is not started')
            residency = (role, model_server_profile)
            transitioned = self._active_role != residency
            if transitioned:
                if self._active_role is not None:
                    await self._release_active_resources()
                    self._active_role = None
                try:
                    await self._activate(role, model_server_profile)
                    self._active_role = residency
                except BaseException:
                    self._active_role = residency
                    try:
                        await self._release_active_resources()
                    finally:
                        self._active_role = None
                    raise
            try:
                return await operation()
            except BaseException:
                if transitioned:
                    try:
                        await self._release_active_resources()
                    finally:
                        self._active_role = None
                raise

    async def _activate(
        self,
        role: ResidentRole,
        model_server_profile: str | None,
    ) -> None:
        if role is ResidentRole.LLM:
            assert model_server_profile is not None
            assert self._router is not None
            await self._router.ensure_model_server(model_server_profile)
        elif role is ResidentRole.EMBEDDING:
            assert self._embedding_server is not None
            await self._embedding_server.start()
        else:
            assert self._reranker_server is not None
            await self._reranker_server.start()

    async def _release_active_resources(self) -> None:
        if self._active_role is None:
            return
        role, _ = self._active_role
        if role is ResidentRole.LLM:
            assert self._router is not None
            await self._router.release_model_server()
        elif role is ResidentRole.EMBEDDING:
            assert self._embedding_server is not None
            await self._embedding_server.stop()
        else:
            assert self._reranker_server is not None
            await self._reranker_server.stop()

    async def _close_resources(self) -> None:
        errors: list[BaseException] = []
        for resource in (
            self._embedding_server,
            self._reranker_server,
        ):
            if resource is not None:
                try:
                    await resource.stop()
                except BaseException as error:
                    errors.append(error)
        for client in (self._embedding_client, self._reranker_client):
            if client is not None:
                try:
                    await client.aclose()
                except BaseException as error:
                    errors.append(error)
        if self._router is not None:
            try:
                await self._router.close()
            except BaseException as error:
                errors.append(error)
        self._embedding_server = None
        self._reranker_server = None
        self._embedding_client = None
        self._reranker_client = None
        self._router = None
        if errors:
            raise errors[0]

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


class RuntimePredictor(dspy.Module):
    """Run one DSPy predictor through the runtime's LLM residency."""

    def __init__(
        self,
        runtime: LocalModelRuntime,
        model_server_profile: str,
        predictor: dspy.Module,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._model_server_profile = model_server_profile
        self.predictor = predictor

    def forward(self, **kwargs: object) -> object:
        """Run synchronous DSPy work outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(**kwargs))
        raise RuntimeError('use acall() from an active event loop')

    async def acall(self, **kwargs: object) -> object:
        """Run asynchronous DSPy work with the requested profile resident."""

        async def operation() -> object:
            return await self.predictor.acall(**kwargs)

        return await self._runtime._execute(
            ResidentRole.LLM,
            self._model_server_profile,
            operation,
        )


__all__ = ['LocalModelRuntime', 'ResidentRole', 'RuntimePredictor']
