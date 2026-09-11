"""Lifecycle-bound DSPy, embedding, and reranking clients."""

import asyncio
from collections.abc import Sequence

import dspy
import httpx

from kms2.config import (
    EmbeddingSettings,
    PredictorStrategy,
    RerankerSettings,
    StageInferenceSettings,
)
from kms2.local_models.coordinator import (
    _GpuCoordinator,
    _GpuResidency,
    _GpuRole,
)
from kms2.local_models.dedicated import _DedicatedRole
from kms2.local_models.router import _RouterServer


class _RouterProfilePredictor(dspy.Module):
    """Run a DSPy predictor with its model-server profile resident."""

    def __init__(
        self,
        router: _RouterServer,
        coordinator: _GpuCoordinator,
        model_server_profile: str,
        predictor: dspy.Module,
    ) -> None:
        super().__init__()
        self._router = router
        self._coordinator = coordinator
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
        """Run asynchronous DSPy work with the selected model server resident."""
        return await self._coordinator.execute(
            _GpuResidency(_GpuRole.LLM, self._model_server_profile),
            lambda: self._router.ensure_model_server(
                self._model_server_profile
            ),
            self._router.release_model_server,
            lambda: self.predictor.acall(**kwargs),
        )


class _EmbeddingClient:
    """Send batched embedding requests through the owned embedding role."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        client: httpx.AsyncClient,
        coordinator: _GpuCoordinator,
        role: _DedicatedRole,
    ) -> None:
        self._settings = settings
        self._client = client
        self._coordinator = coordinator
        self._role = role

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed inputs in configured batches, preserving server order."""
        values = list(texts)
        return await self._coordinator.execute(
            _GpuResidency(_GpuRole.EMBEDDING),
            self._role.start,
            self._role.close,
            lambda: self._embed(values),
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._settings.batch_size):
            response = await self._client.post(
                '/v1/embeddings',
                json={
                    'model': self._settings.model.model_id,
                    'input': texts[start : start + self._settings.batch_size],
                },
            )
            response.raise_for_status()
            vectors.extend(
                item['embedding'] for item in response.json()['data']
            )
        return vectors


class _RerankerClient:
    """Send reranking requests through the owned reranker role."""

    def __init__(
        self,
        settings: RerankerSettings,
        client: httpx.AsyncClient,
        coordinator: _GpuCoordinator,
        role: _DedicatedRole,
    ) -> None:
        self._settings = settings
        self._client = client
        self._coordinator = coordinator
        self._role = role

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[dict[str, object]]:
        """Return the reranking endpoint payload unchanged."""
        return await self._coordinator.execute(
            _GpuResidency(_GpuRole.RERANKER),
            self._role.start,
            self._role.close,
            lambda: self._rerank(query, documents, top_n),
        )

    async def _rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None,
    ) -> list[dict[str, object]]:
        body: dict[str, object] = {
            'model': self._settings.model.model_id,
            'query': query,
            'documents': list(documents),
        }
        if top_n is not None:
            body['top_n'] = top_n
        response = await self._client.post('/v1/rerank', json=body)
        response.raise_for_status()
        return response.json()['results']


def _predictor(
    router: _RouterServer,
    coordinator: _GpuCoordinator,
    inference: StageInferenceSettings,
    signature: type[dspy.Signature],
) -> _RouterProfilePredictor:
    """Build a DSPy predictor for one model-server profile."""
    lm = dspy.LM(
        inference.model_server_profile,
        api_base=f'{router.endpoint}/v1',
        api_key='not-needed',
        custom_llm_provider='openai',
        temperature=inference.temperature,
        max_tokens=inference.max_tokens,
        num_retries=inference.num_retries,
        cache=inference.cache,
    )
    predictor: dspy.Module
    if inference.strategy is PredictorStrategy.PREDICT:
        predictor = dspy.Predict(signature)
    else:
        predictor = dspy.ChainOfThought(signature)
    predictor.set_lm(lm)
    return _RouterProfilePredictor(
        router,
        coordinator,
        inference.model_server_profile,
        predictor,
    )
