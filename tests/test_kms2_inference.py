import asyncio

import dspy
import pytest

from kms2.config import PredictorStrategy, StageInferenceSettings
from kms2.local_models.coordinator import _GpuCoordinator
from kms2.local_models.inference import _predictor, _RouterProfilePredictor


class _RecordingLM:
    calls: list[tuple[str, dict[str, object]]]

    def __init__(self, model: str, **kwargs: object) -> None:
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


class _Coordinator:
    async def execute(self, role, activate, deactivate, operation):
        del role, activate, deactivate
        return await operation()


class _ModelServerRouter:
    endpoint = 'http://router'

    def __init__(self) -> None:
        self.ensured_model_servers: list[str] = []

    async def ensure_model_server(self, model_server_profile: str) -> None:
        self.ensured_model_servers.append(model_server_profile)

    async def release_model_server(self) -> None:
        return None


class _AsyncPredictor:
    async def acall(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)


def _stage_inference(
    strategy: PredictorStrategy,
) -> StageInferenceSettings:
    return StageInferenceSettings(
        model_server_profile='text',
        strategy=strategy,
        temperature=0.2,
        max_tokens=123,
        num_retries=4,
    )


def test_local_model_predictor_forwards_inference_options(monkeypatch):
    _RecordingLM.calls = []
    monkeypatch.setattr('kms2.local_models.inference.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms2.local_models.inference.dspy.Predict', _RecordingPredictor
    )

    predictor = _predictor(
        _Router(),
        _Coordinator(),
        _stage_inference(PredictorStrategy.PREDICT),
        _Signature,
    )

    assert isinstance(predictor.predictor, _RecordingPredictor)
    assert _RecordingLM.calls == [
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


def test_local_model_predictor_supports_configured_strategies(monkeypatch):
    _RecordingLM.calls = []
    monkeypatch.setattr('kms2.local_models.inference.dspy.LM', _RecordingLM)
    monkeypatch.setattr(
        'kms2.local_models.inference.dspy.Predict', _RecordingPredictor
    )
    monkeypatch.setattr(
        'kms2.local_models.inference.dspy.ChainOfThought',
        _RecordingChainOfThought,
    )

    predictor = _predictor(
        _Router(),
        _Coordinator(),
        _stage_inference(PredictorStrategy.CHAIN_OF_THOUGHT),
        _Signature,
    )

    assert isinstance(predictor.predictor, _RecordingChainOfThought)


def test_stage_inference_rejects_cache_override():
    with pytest.raises(ValueError):
        StageInferenceSettings(model_server_profile='text', cache=True)


def test_router_profile_predictor_switches_model_server_profiles():
    router = _ModelServerRouter()
    coordinator = _GpuCoordinator()
    text = _RouterProfilePredictor(
        router,
        coordinator,
        'text',
        _AsyncPredictor(),
    )
    vision = _RouterProfilePredictor(
        router,
        coordinator,
        'vision',
        _AsyncPredictor(),
    )

    async def run_predictions() -> None:
        assert await text.acall(content='first') == {'content': 'first'}
        assert await text.acall(content='second') == {'content': 'second'}
        assert await vision.acall(content='third') == {'content': 'third'}

    asyncio.run(run_predictions())

    assert router.ensured_model_servers == ['text', 'vision']
