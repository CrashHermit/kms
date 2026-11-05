import asyncio

import dspy

from kms.composition.predictors import PredictorFactory
from kms.config.inference import (
    PredictorStrategy,
    StageInferenceSettings,
    TextInferenceSettings,
)
from kms.config.llm import ChatGPTLLMProfile, LLMSettings
from kms.config.runtime import LocalModelRuntimeSettings
from kms.config.settings import Settings
from kms.local_models.runtime import RuntimePredictor


class _RecordingLM:
    calls: list[tuple[str, dict[str, object]]]

    def __init__(self, model: str, **kwargs: object):
        self.calls.append((model, kwargs))


class _RecordingPredictor:
    def __init__(self, signature: type[dspy.Signature]) -> None:
        self.signature = signature
        self.lm = None

    def set_lm(self, lm: object) -> None:
        self.lm = lm


class _RecordingChainOfThought(_RecordingPredictor):
    pass


class _Signature(dspy.Signature):
    pass


class _Router:
    endpoint = 'http://router'


class _AsyncPredictor:
    async def acall(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)


def _stage_inference(strategy: PredictorStrategy) -> StageInferenceSettings:
    return StageInferenceSettings(
        profile='qwen3.8-9b-distill-text',
        strategy=strategy,
        temperature=0.2,
        top_p=0.8,
        top_k=12,
        max_tokens=123,
        num_retries=4,
    )


def test_inference_defaults_match_model_recommendations():
    settings = StageInferenceSettings(profile='qwen3.8-9b-distill-text')
    text_settings = TextInferenceSettings()

    assert settings.temperature == 1.0
    assert settings.top_p == 0.95
    assert settings.top_k == 64
    assert settings.max_tokens == 8192
    assert text_settings.profile is None
    assert text_settings.temperature == 0.6
    assert text_settings.top_p == 0.95
    assert text_settings.top_k == 20
    assert text_settings.max_tokens == 16384


def test_runtime_predictor_supports_configured_strategies(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    _RecordingLM.calls = calls
    monkeypatch.setattr('kms.local_models.runtime.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms.local_models.runtime.dspy.ChainOfThought',
        _RecordingChainOfThought,
    )
    from kms.local_models.runtime import LocalModelRuntime

    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
    runtime._router = _Router()
    predictor = runtime.predictor(
        _stage_inference(PredictorStrategy.CHAIN_OF_THOUGHT),
        _Signature,
        model_server_profile='qwen3.8-9b-distill-text',
    )
    assert isinstance(predictor.predictor, _RecordingChainOfThought)


def test_runtime_predictor_executes_through_requested_residency():
    asyncio.run(_test_runtime_predictor_executes_through_requested_residency())


async def _test_runtime_predictor_executes_through_requested_residency():
    calls: list[str] = []

    class Runtime:
        async def _execute(self, model_server_profile, operation):
            calls.append(model_server_profile)
            return await operation()

    text = RuntimePredictor(Runtime(), 'text', _AsyncPredictor())
    vision = RuntimePredictor(Runtime(), 'vision', _AsyncPredictor())

    assert await text.acall(content='first') == {'content': 'first'}
    assert await text.acall(content='second') == {'content': 'second'}
    assert await vision.acall(content='third') == {'content': 'third'}
    assert calls == ['text', 'text', 'vision']


def test_strict_adapter_scope_is_local_to_its_predictor():
    asyncio.run(_test_strict_adapter_scope_is_local_to_its_predictor())


async def _test_strict_adapter_scope_is_local_to_its_predictor():
    strict = dspy.ChatAdapter(use_json_adapter_fallback=False)
    ambient = dspy.JSONAdapter()
    observed = []
    strict_running = asyncio.Event()
    release_strict = asyncio.Event()

    class Runtime:
        async def _execute(self, _model_server_profile, operation):
            return await operation()

    class Predictor:
        async def acall(self, *, wait=False):
            observed.append(dspy.settings.adapter)
            if wait:
                strict_running.set()
                await release_strict.wait()
            return 'ok'

    bounded = RuntimePredictor(
        Runtime(), 'bounded', Predictor(), adapter=strict
    )
    ordinary = RuntimePredictor(Runtime(), 'ordinary', Predictor())
    with dspy.context(adapter=ambient):
        task = asyncio.create_task(bounded.acall(wait=True))
        await strict_running.wait()
        assert await ordinary.acall() == 'ok'
        release_strict.set()
        assert await task == 'ok'
        assert dspy.settings.adapter is ambient
    assert observed == [strict, ambient]


class _CloudPredictor(dspy.Module):
    active_count = 0
    peak_count = 0

    def __init__(self, signature: type[dspy.Signature]) -> None:
        super().__init__()
        self.signature = signature
        self.lm = None

    def set_lm(self, lm: object) -> None:
        self.lm = lm

    async def aforward(self, **kwargs: object) -> dict[str, object]:
        type(self).active_count += 1
        type(self).peak_count = max(
            type(self).peak_count, type(self).active_count
        )
        try:
            await asyncio.sleep(0.01)
            return dict(kwargs)
        finally:
            type(self).active_count -= 1


def test_chatgpt_predictor_uses_native_responses_without_local_options(
    monkeypatch,
):
    _CloudPredictor.active_count = 0
    _CloudPredictor.peak_count = 0
    calls: list[tuple[str, dict[str, object]]] = []
    _RecordingLM.calls = calls
    cloud_predictors: list[_CloudPredictor] = []

    def create_predictor(signature: type[dspy.Signature]) -> _CloudPredictor:
        predictor = _CloudPredictor(signature)
        cloud_predictors.append(predictor)
        return predictor

    monkeypatch.setattr('kms.composition.predictors.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms.composition.predictors.dspy.Predict', create_predictor
    )

    class LocalModels:
        def predictor(self, *args: object, **kwargs: object) -> object:
            raise AssertionError('remote profile touched local models')

    settings = Settings(
        llm=LLMSettings(
            max_concurrency=2,
            default_text_profile='cloud',
            profiles={'cloud': ChatGPTLLMProfile()},
        )
    )
    factory = PredictorFactory(settings, LocalModels(), None)
    inference = StageInferenceSettings(
        strategy=PredictorStrategy.PREDICT,
        temperature=0.1,
        top_p=0.2,
        top_k=3,
        max_tokens=99,
        num_retries=0,
    )
    first = factory.create(_CloudPredictor, inference, _Signature)
    second = factory.create(_CloudPredictor, inference, _Signature)
    third = factory.create(_CloudPredictor, inference, _Signature)
    fourth = factory.create(_CloudPredictor, inference, _Signature)
    asyncio.run(
        _assert_cloud_predictors_are_bounded(first, second, third, fourth)
    )


async def _assert_cloud_predictors_are_bounded(
    *predictors: dspy.Module,
) -> None:
    await asyncio.gather(
        *(
            predictor.acall(value=str(index))
            for index, predictor in enumerate(predictors)
        )
    )
    assert _CloudPredictor.peak_count == 2
