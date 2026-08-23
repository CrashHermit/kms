import asyncio
from types import SimpleNamespace

import pytest

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


class _Recorder:
    def __init__(self):
        self.calls = []

    def record(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_as_list_normalizes_dspy_cardinality():
    item = object()
    assert module.as_list(None) == []
    assert module.as_list(item) == [item]
    assert module.as_list((1, 2)) == [1, 2]
    assert module.as_list([1, 2]) == [1, 2]


def test_require_bool_rejects_coercible_values():
    assert module.require_bool(True, 'decision') is True
    with pytest.raises(ValueError, match='decision must be a boolean'):
        module.require_bool('true', 'decision')


def test_require_text_rejects_empty_values():
    assert module.require_text(' answer ', 'result') == ' answer '
    with pytest.raises(ValueError, match='result must be a non-empty string'):
        module.require_text('', 'result')
    with pytest.raises(ValueError, match='result must be a non-empty string'):
        module.require_text(None, 'result')


def test_require_number_validates_type_and_bounds():
    assert module.require_number(0.5, 'confidence', minimum=0, maximum=1) == 0.5
    with pytest.raises(TypeError, match='confidence must be a number'):
        module.require_number(True, 'confidence')
    with pytest.raises(ValueError, match='at most 1'):
        module.require_number(1.1, 'confidence', maximum=1)
    with pytest.raises(ValueError, match='finite'):
        module.require_number(float('inf'), 'confidence')


def test_require_positions_validates_local_contract():
    assert module.require_positions(
        [0, 2], field_name='positions', upper_bound=3, ordered=True
    ) == [0, 2]
    with pytest.raises(TypeError, match=r'positions\[0\] must be an int'):
        module.require_positions(
            ['0'], field_name='positions', upper_bound=1
        )
    with pytest.raises(ValueError, match='outside'):
        module.require_positions(
            [3], field_name='positions', upper_bound=3
        )
    with pytest.raises(ValueError, match='duplicate'):
        module.require_positions(
            [1, 1], field_name='positions', upper_bound=2
        )
    with pytest.raises(ValueError, match='document order'):
        module.require_positions(
            [1, 0], field_name='positions', upper_bound=2, ordered=True
        )


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


def test_module_records_only_after_decode_succeeds(monkeypatch):
    language_model = SimpleNamespace(model='openai/source-model')
    recorder = _Recorder()
    instance = _Module(language_model, recorder)
    instance.predictor = _Predictor()
    monkeypatch.setattr(module.serve, 'current_model_manager', lambda: None)

    assert asyncio.run(instance.aforward(value='one')) == 'result'
    assert len(recorder.calls) == 1

    class _InvalidModule(_Module):
        def decode(self, prediction, **inputs):
            raise ValueError('invalid prediction')

    invalid_recorder = _Recorder()
    invalid = _InvalidModule(language_model, invalid_recorder)
    invalid.predictor = _Predictor()
    with pytest.raises(ValueError, match='invalid prediction'):
        asyncio.run(invalid.aforward(value='one'))
    assert invalid_recorder.calls == []
