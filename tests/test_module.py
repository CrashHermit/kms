import asyncio
from types import SimpleNamespace

from kms.core import module


class _Module(module.Module):
    signature = object

    def encode(self, value: str) -> dict:
        return {'value': value}

    def decode(self, prediction, **inputs) -> str:
        return prediction.value


class _Predictor:
    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value='result')


class _Manager:
    def __init__(self):
        self.calls = []

    async def aexecute(self, model_id, operation):
        self.calls.append(model_id)
        return await operation()


def test_module_resolves_serving_model_for_each_call(monkeypatch):
    language_model = SimpleNamespace(
        model='openai/source-model', _kms_module_name='stage'
    )
    instance = _Module(language_model)
    instance.predictor = _Predictor()
    manager = _Manager()
    configured_models = iter(['first-preset', 'second-preset'])

    monkeypatch.setattr(module.serve, 'current_model_manager', lambda: manager)
    monkeypatch.setattr(
        module.llm,
        'configured_serving_model',
        lambda name: next(configured_models),
    )

    async def run():
        first = await instance.aforward(value='one')
        second = await instance.aforward(value='two')
        return first, second

    assert asyncio.run(run()) == ('result', 'result')
    assert manager.calls == ['first-preset', 'second-preset']


def test_module_without_manager_calls_predictor_directly(monkeypatch):
    language_model = SimpleNamespace(model='openai/source-model')
    instance = _Module(language_model)
    instance.predictor = _Predictor()
    monkeypatch.setattr(module.serve, 'current_model_manager', lambda: None)

    assert asyncio.run(instance.aforward(value='one')) == 'result'
    assert instance.predictor.calls == [{'value': 'one'}]
