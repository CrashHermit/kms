import asyncio

import pytest

from kms import config
from kms.core import serve


def _config(**kw):
    serving = config.load_settings().serving
    defaults = dict(
        start=['llama-server', '--port', str(serving.port)],
        endpoint=f'http://{serving.host}:{serving.port}',
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


def test_ensure_model_rejects_router_without_target_model(monkeypatch):
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('qwen3.5-9b-text', 'unloaded')),
    )
    manager = serve.RouterManager(_config())
    with pytest.raises(RuntimeError, match='does not expose model'):
        manager.ensure_model('qwen3-vl-4b')


def test_ensure_model_unloads_others_and_loads_target(monkeypatch):
    calls = []
    loaded = 'qwen3.5-9b-text'

    def fake_get(url, timeout):
        ids = ('qwen3.5-9b-text', 'qwen3-vl-4b')
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
    manager.ensure_model('qwen3-vl-4b')
    assert (
        'http://127.0.0.1:8080/models/unload',
        {'model': 'qwen3.5-9b-text'},
    ) in calls
    assert (
        'http://127.0.0.1:8080/models/load',
        {'model': 'qwen3-vl-4b'},
    ) in calls


def test_ensure_model_skips_load_when_already_loaded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('qwen3.5-9b-text', 'loaded')),
    )
    monkeypatch.setattr(
        serve,
        '_post_json',
        lambda url, body, timeout: calls.append((url, body)),
    )
    manager = serve.RouterManager(_config())
    manager.ensure_model('qwen3.5-9b-text')
    assert calls == []


def test_ensure_model_records_resident_model(monkeypatch):
    monkeypatch.setattr(
        serve,
        '_get_json',
        lambda url, timeout: _statuses(('model-a', 'loaded')),
    )
    manager = serve.RouterManager(_config())
    manager.ensure_model('model-a')
    manager.ensure_model('model-a')
    assert manager.resident == 'model-a'


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


def test_aexecute_uses_operation_lease():
    manager = serve.RouterManager(_config())
    calls = []

    async def fake_ensure(model_id):
        calls.append(('ensure', model_id))
        manager._resident = model_id

    async def operation():
        calls.append(('operation', manager.resident))
        return 'done'

    manager.aensure_model = fake_ensure
    assert asyncio.run(manager.aexecute('model-a', operation)) == 'done'
    assert calls == [('ensure', 'model-a'), ('operation', 'model-a')]


def test_preset_ini_contains_configured_models(tmp_path):
    ini = serve._preset_ini(tmp_path)
    assert '[gemma-4-e4b-qat-text]' in ini
    assert 'mmproj' in ini
    assert 'ctx-size = 32768' in ini


def test_default_router_uses_configured_preset_file(monkeypatch, tmp_path):
    monkeypatch.setattr(serve.Path, 'home', classmethod(lambda cls: tmp_path))
    config = serve.default_router()
    assert '--models-preset' in config.start
    assert '--models-max' in config.start
    assert (tmp_path / 'models' / 'kms-models.ini').exists()


def test_default_retrieval_servers_use_configured_devices_and_batch_sizes():
    embedding = serve.default_embedding_server()._config
    reranker = serve.default_reranker_server()._config
    assert embedding.endpoint.endswith(':8081')
    assert reranker.endpoint.endswith(':8082')
    assert '--n-gpu-layers' in embedding.start
    assert embedding.start[embedding.start.index('--n-gpu-layers') + 1] == '999'
    assert '--device' in embedding.start
    assert embedding.start[embedding.start.index('--device') + 1] == 'CUDA0'
    assert '--device' in reranker.start
    assert reranker.start[reranker.start.index('--device') + 1] == 'CUDA0'
    assert '--reranking' in reranker.start
    assert reranker.start[reranker.start.index('--ubatch-size') + 1] == '1024'


def test_dedicated_server_reports_missing_model(monkeypatch, tmp_path):
    server = serve.DedicatedServer(
        serve.DedicatedServerConfig(
            start=['llama-server', '--model', str(tmp_path / 'missing.gguf')],
            endpoint='http://127.0.0.1:9999',
            ready_timeout=1.0,
            poll_interval=0.0,
            request_timeout=1.0,
            terminate_timeout=1.0,
        )
    )
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: None)
    with pytest.raises(RuntimeError, match='missing.gguf'):
        server.ensure_started()


def test_dedicated_server_waits_for_health(monkeypatch, tmp_path):
    model = tmp_path / 'model.gguf'
    model.touch()
    states = iter([None, {'status': 'ok'}])
    pops = []
    monkeypatch.setattr(serve, '_get_json', lambda url, timeout: next(states))
    monkeypatch.setattr(
        serve.subprocess,
        'Popen',
        lambda *args, **kwargs: pops.append((args, kwargs)) or _FakeProc(),
    )
    server = serve.DedicatedServer(
        serve.DedicatedServerConfig(
            start=['llama-server', '--model', str(model)],
            endpoint='http://127.0.0.1:9999',
            ready_timeout=1.0,
            poll_interval=0.0,
            request_timeout=1.0,
            terminate_timeout=1.0,
        )
    )
    server.ensure_started()
    assert pops[0][1]['start_new_session'] is True
