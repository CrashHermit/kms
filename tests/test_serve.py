import pytest

from kms import config
from kms.core import serve


def _config(**kw):
    serving = config.load_settings().serving
    defaults = dict(
        start=['llama-server', '--port', str(serving.port)],
        endpoint=f'http://{serving.host}:{serving.port}',
        models={'formatter': 'qwen3.5-9b', 'corrector': 'qwen3-vl-4b'},
        ready_timeout=serving.ready_timeout,
        poll_interval=serving.poll_interval,
        request_timeout=serving.request_timeout,
        terminate_timeout=serving.terminate_timeout,
    )
    defaults.update(kw)
    return serve.RouterConfig(**defaults)


def _statuses(*pairs):
    return {
        'data': [
            {'id': mid, 'status': {'value': status}} for mid, status in pairs
        ]
    }


class _FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_switch_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: _statuses())
    manager = serve.RouterManager(_config())
    with pytest.raises(RuntimeError, match='unknown model'):
        manager.switch('nope')


def test_switch_rejects_router_without_target_model(monkeypatch):
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('qwen3.5-9b', 'unloaded')),
    )
    manager = serve.RouterManager(_config())
    with pytest.raises(RuntimeError, match='does not expose model'):
        manager.switch('corrector')


def test_switch_unloads_others_and_loads_target(monkeypatch):
    calls = []
    loaded = 'qwen3.5-9b'

    def fake_get(url, timeout):
        ids = ('qwen3.5-9b', 'qwen3-vl-4b')
        return _statuses(
            *((mid, 'loaded' if mid == loaded else 'unloaded') for mid in ids)
        )

    def fake_post(url, body, timeout):
        nonlocal loaded
        calls.append((url, body))
        if url.endswith('/models/load'):
            loaded = body['model']

    monkeypatch.setattr(serve, '_get_json', fake_get)
    monkeypatch.setattr(serve, '_post_json', fake_post)
    manager = serve.RouterManager(_config())
    manager.switch('corrector')
    assert (
        'http://127.0.0.1:8080/models/unload',
        {'model': 'qwen3.5-9b'},
    ) in calls
    assert (
        'http://127.0.0.1:8080/models/load',
        {'model': 'qwen3-vl-4b'},
    ) in calls


def test_switch_skips_load_when_already_loaded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('qwen3.5-9b', 'loaded')),
    )
    monkeypatch.setattr(
        serve,
        '_post_json',
        lambda url, body, timeout: calls.append((url, body)),
    )
    manager = serve.RouterManager(_config())
    manager.switch('formatter')
    assert calls == []


def test_wait_loaded_raises_on_failed(monkeypatch):
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('qwen3-vl-4b', 'failed')),
    )
    manager = serve.RouterManager(_config())
    with pytest.raises(RuntimeError, match='failed to load'):
        manager._wait_loaded('qwen3-vl-4b')


def test_wait_loaded_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('qwen3-vl-4b', 'loading')),
    )
    manager = serve.RouterManager(_config(ready_timeout=0.0, poll_interval=0.0))
    with pytest.raises(RuntimeError, match='not loaded'):
        manager._wait_loaded('qwen3-vl-4b')


def test_ensure_router_starts_when_down(monkeypatch):
    states = iter([None, _statuses()])
    pops = []
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: next(states))
    monkeypatch.setattr(
        serve.subprocess,
        'Popen',
        lambda *a, **kw: pops.append(a) or _FakeProc(),
    )
    manager = serve.RouterManager(_config())
    manager._ensure_router()
    assert pops


def test_ensure_router_skips_when_up(monkeypatch):
    pops = []
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: _statuses())
    monkeypatch.setattr(
        serve.subprocess,
        'Popen',
        lambda *a, **kw: pops.append(a) or _FakeProc(),
    )
    manager = serve.RouterManager(_config())
    manager._ensure_router()
    assert pops == []


def test_ensure_router_raises_on_early_exit(monkeypatch):
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: None)
    monkeypatch.setattr(
        serve.subprocess, 'Popen', lambda *a, **kw: _FakeProc(returncode=1)
    )
    manager = serve.RouterManager(_config())
    with pytest.raises(RuntimeError, match='exited early'):
        manager._ensure_router()


def test_ensure_router_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: None)
    monkeypatch.setattr(serve.subprocess, 'Popen', lambda *a, **kw: _FakeProc())
    manager = serve.RouterManager(_config(ready_timeout=0.0, poll_interval=0.0))
    with pytest.raises(RuntimeError, match='not ready'):
        manager._ensure_router()


def test_preset_ini_contains_both_models(tmp_path):
    ini = serve._preset_ini(tmp_path)
    assert '[qwen3.5-9b]' in ini
    assert '[qwen3-vl-4b]' in ini
    assert 'mmproj' in ini
    assert 'ctx-size = 32768' in ini


def test_default_router_maps_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(serve.Path, 'home', classmethod(lambda cls: tmp_path))
    config = serve.default_router()
    assert config.models['formatter'] == 'qwen3.5-9b'
    assert config.models['corrector'] == 'qwen3.5-9b'
    assert 'pipeline' not in config.models
    assert '--models-preset' in config.start
    assert '--models-max' in config.start
    assert (tmp_path / 'models' / 'kms-models.ini').exists()
