"""Public lifecycle root for KMS2 local model services."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from types import TracebackType

import dspy

from kms.config.inference import PredictorStrategy, StageInferenceSettings
from kms.config.runtime import LocalModelRuntimeSettings
from kms.local_models.ownership import (
    ProcessOwner,
    create_process_owner,
)
from kms.local_models.router import RouterServer

Operation = Callable[[], Awaitable[object]]


class LocalModelRuntime:
    """Own the local LLM router and its active model profile."""

    def __init__(self, settings: LocalModelRuntimeSettings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._router_started = False
        self._state = 'stopped'
        self._active_profile: str | None = None
        self._process_owner: ProcessOwner = create_process_owner()
        self._router: RouterServer | None = None

    async def start(self) -> None:
        """Prepare the router without starting its model process."""
        async with self._lock:
            if self._state != 'stopped':
                raise RuntimeError('local model runtime is already started')
            self._state = 'starting'
            api_prefix = f'/kms-{uuid.uuid4().hex}'
            self._router = RouterServer(
                self._settings.router,
                api_prefix,
                self._process_owner,
            )
            self._state = 'started'

    async def close(self) -> None:
        """Release the owned router idempotently."""
        async with self._lock:
            if self._state == 'stopped':
                return
            self._state = 'closing'
            self._active_profile = None
            try:
                await self._close_resources()
            finally:
                self._state = 'stopped'

    def predictor(
        self,
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
        *,
        model_server_profile: str,
        adapter: dspy.Adapter | None = None,
        examples: Sequence[dspy.Example] | None = None,
    ) -> RuntimePredictor:
        """Build a predictor with explicit examples before the residency wrapper."""
        router = self._router
        if router is None:
            raise RuntimeError('local model runtime has not started')
        lm = dspy.LM(
            model_server_profile,
            api_base=f'{router.endpoint}/v1',
            api_key='not-needed',
            custom_llm_provider='openai',
            temperature=inference.temperature,
            top_p=inference.top_p,
            top_k=inference.top_k,
            max_tokens=inference.max_tokens,
            num_retries=inference.num_retries,
            cache=True,
        )
        if inference.strategy is PredictorStrategy.PREDICT:
            predictor = dspy.Predict(signature)
        else:
            predictor = dspy.ChainOfThought(signature)
        if examples is not None:
            for parameter in predictor.predictors():
                parameter.demos = list(examples)
        predictor.set_lm(lm)
        return RuntimePredictor(
            self, model_server_profile, predictor, adapter=adapter
        )

    async def _execute(
        self,
        model_server_profile: str,
        operation: Operation,
    ) -> object:
        async with self._lock:
            if self._state != 'started':
                raise RuntimeError('local model runtime is not started')
            transitioned = self._active_profile != model_server_profile
            if transitioned:
                if self._active_profile is not None:
                    await self._release_active_resources()
                    self._active_profile = None
                try:
                    await self._activate(model_server_profile)
                    self._active_profile = model_server_profile
                except BaseException:
                    self._active_profile = model_server_profile
                    try:
                        await self._release_active_resources()
                    finally:
                        self._active_profile = None
                    raise
            try:
                return await operation()
            except BaseException:
                if transitioned:
                    try:
                        await self._release_active_resources()
                    finally:
                        self._active_profile = None
                raise

    async def _activate(self, model_server_profile: str) -> None:
        assert self._router is not None
        if not self._router_started:
            await self._router.start()
            self._router_started = True
        await self._router.ensure_model_server(model_server_profile)

    async def _release_active_resources(self) -> None:
        if self._active_profile is not None and self._router_started:
            assert self._router is not None
            await self._router.release_model_server()

    async def _close_resources(self) -> None:
        errors: list[BaseException] = []
        if self._router is not None:
            try:
                await self._router.close()
            except BaseException as error:
                errors.append(error)
        self._router_started = False
        self._router = None
        if errors:
            raise errors[0]

    async def __aenter__(self) -> LocalModelRuntime:
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
        *,
        adapter: dspy.Adapter | None = None,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._model_server_profile = model_server_profile
        self.predictor = predictor
        self._adapter = adapter

    def forward(self, **kwargs: object) -> object:
        """Run synchronous DSPy work outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(**kwargs))
        raise RuntimeError('use acall() from an active event loop')

    async def aforward(self, **kwargs: object) -> object:
        """Run asynchronous DSPy work with the requested profile resident."""

        async def operation() -> object:
            if self._adapter is None:
                return await self.predictor.acall(**kwargs)
            with dspy.context(adapter=self._adapter):
                return await self.predictor.acall(**kwargs)

        return await self._runtime._execute(
            self._model_server_profile,
            operation,
        )
