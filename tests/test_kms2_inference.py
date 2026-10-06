import asyncio

import dspy

from kms2.config.inference import (
    PredictorStrategy,
    StageInferenceSettings,
    TextInferenceSettings,
)
from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.local_models.runtime import ResidentRole, RuntimePredictor


class _EmbeddingCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1] * len(texts)


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
        model_server_profile='text',
        strategy=strategy,
        temperature=0.2,
        top_p=0.8,
        top_k=12,
        max_tokens=123,
        num_retries=4,
    )


def test_inference_defaults_match_model_recommendations():
    settings = StageInferenceSettings(model_server_profile='text')
    text_settings = TextInferenceSettings()

    assert settings.temperature == 1.0
    assert settings.top_p == 0.95
    assert settings.top_k == 64
    assert settings.max_tokens == 8192
    assert text_settings.model_server_profile == 'qwen3.8-9b-distill-text'
    assert text_settings.temperature == 0.6
    assert text_settings.top_p == 0.95
    assert text_settings.top_k == 20
    assert text_settings.max_tokens == 16384


def test_runtime_predictor_forwards_inference_options(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    _RecordingLM.calls = calls
    monkeypatch.setattr('kms2.local_models.runtime.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms2.local_models.runtime.dspy.Predict', _RecordingPredictor
    )

    from kms2.local_models.runtime import LocalModelRuntime

    runtime = LocalModelRuntime(
        LocalModelRuntimeSettings(),
        embedding_token_counter=_EmbeddingCounter(),
    )
    runtime._router = _Router()
    predictor = runtime.predictor(
        _stage_inference(PredictorStrategy.PREDICT),
        _Signature,
    )

    assert isinstance(predictor, RuntimePredictor)
    assert calls == [
        (
            'text',
            {
                'api_base': 'http://router/v1',
                'api_key': 'not-needed',
                'custom_llm_provider': 'openai',
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 12,
                'max_tokens': 123,
                'num_retries': 4,
                'cache': True,
            },
        )
    ]


def test_runtime_predictor_supports_configured_strategies(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    _RecordingLM.calls = calls
    monkeypatch.setattr('kms2.local_models.runtime.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms2.local_models.runtime.dspy.ChainOfThought',
        _RecordingChainOfThought,
    )
    from kms2.local_models.runtime import LocalModelRuntime

    runtime = LocalModelRuntime(
        LocalModelRuntimeSettings(),
        embedding_token_counter=_EmbeddingCounter(),
    )
    runtime._router = _Router()
    predictor = runtime.predictor(
        _stage_inference(PredictorStrategy.CHAIN_OF_THOUGHT),
        _Signature,
    )

    assert isinstance(predictor.predictor, _RecordingChainOfThought)


def test_runtime_predictor_executes_through_requested_residency():
    asyncio.run(_test_runtime_predictor_executes_through_requested_residency())


async def _test_runtime_predictor_executes_through_requested_residency():
    calls: list[tuple[ResidentRole, str | None]] = []

    class Runtime:
        async def _execute(self, role, profile, operation):
            calls.append((role, profile))
            return await operation()

    text = RuntimePredictor(Runtime(), 'text', _AsyncPredictor())
    vision = RuntimePredictor(Runtime(), 'vision', _AsyncPredictor())

    assert await text.acall(content='first') == {'content': 'first'}
    assert await text.acall(content='second') == {'content': 'second'}
    assert await vision.acall(content='third') == {'content': 'third'}
    assert calls == [
        (ResidentRole.LLM, 'text'),
        (ResidentRole.LLM, 'text'),
        (ResidentRole.LLM, 'vision'),
    ]


def test_strict_adapter_scope_is_local_to_its_predictor():
    asyncio.run(_test_strict_adapter_scope_is_local_to_its_predictor())


async def _test_strict_adapter_scope_is_local_to_its_predictor():
    strict = dspy.ChatAdapter(use_json_adapter_fallback=False)
    ambient = dspy.JSONAdapter()
    observed = []
    strict_running = asyncio.Event()
    release_strict = asyncio.Event()

    class Runtime:
        async def _execute(self, role, profile, operation):
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
