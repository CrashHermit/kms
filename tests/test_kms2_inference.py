import asyncio

import dspy

from kms2.config.inference import PredictorStrategy, StageInferenceSettings
from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.local_models.runtime import ResidentRole, RuntimePredictor


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
        max_tokens=123,
        num_retries=4,
    )


def test_runtime_predictor_forwards_inference_options(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []
    _RecordingLM.calls = calls
    monkeypatch.setattr('kms2.local_models.runtime.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms2.local_models.runtime.dspy.Predict', _RecordingPredictor
    )

    from kms2.local_models import LocalModelRuntime

    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
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
    from kms2.local_models import LocalModelRuntime

    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
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
