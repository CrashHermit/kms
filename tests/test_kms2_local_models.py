import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from kms2.config.runtime import LlamaServerSettings, LocalModelRuntimeSettings
from kms2.local_models import ownership, posix_ownership
from kms2.local_models.ownership import create_process_owner
from kms2.local_models.posix_ownership import PosixProcessOwner
from kms2.local_models.process import (
    LocalModelPortInUse,
    LocalModelStartError,
    ManagedLlamaServer,
)
from kms2.local_models.router import _model_server_presets_ini, _router_command
from kms2.local_models.runtime import (
    LocalModelRuntime,
    ResidentRole,
    _embedding_command,
    _reranker_command,
)


def test_local_server_commands_have_isolated_api_prefixes():
    settings = LocalModelRuntimeSettings()

    router = _router_command(
        settings.router, Path('/tmp/models.ini'), '/kms2-test'
    )
    embedding = _embedding_command(settings.embedding, '/kms2-test')
    reranker = _reranker_command(settings.reranker, '/kms2-test')

    assert router[-5:] == [
        '--models-preset',
        '/tmp/models.ini',
        '--models-max',
        '1',
        '--no-models-autoload',
    ]
    assert embedding[embedding.index('--api-prefix') + 1] == '/kms2-test'
    assert embedding[embedding.index('--pooling') + 1] == 'last'
    assert '--reranking' in reranker
    assert reranker[reranker.index('--pooling') + 1] == 'rank'
    assert reranker[reranker.index('--ubatch-size') + 1] == '2048'


def test_router_preset_contains_text_and_vision_model_server_profiles():
    preset = _model_server_presets_ini(LocalModelRuntimeSettings().router)

    assert '[gemma-text-32k]' in preset
    assert '[gemma-vision-8k]' in preset
    assert 'ctx-size = 32768' in preset
    assert 'reasoning = off' in preset
    assert 'no-warmup = true' in preset
    assert 'mmproj = ' in preset


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def _server_command(port: int) -> list[str]:
    script = """
import http.server
import sys

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, format, *args):
        pass

http.server.ThreadingHTTPServer(
    ('127.0.0.1', int(sys.argv[1])), Handler
).serve_forever()
"""
    return [sys.executable, '-c', script, str(port)]


def _server_settings(port: int) -> LlamaServerSettings:
    return LlamaServerSettings(
        port=port,
        ready_timeout=3.0,
        poll_interval=0.01,
        health_timeout=0.05,
        terminate_timeout=0.1,
    )


def test_managed_server_starts_and_reaps_owned_process():
    asyncio.run(_test_managed_server_starts_and_reaps_owned_process())


async def _test_managed_server_starts_and_reaps_owned_process():
    port = _free_port()
    server = ManagedLlamaServer(
        _server_command(port),
        f'http://127.0.0.1:{port}',
        '/health',
        _server_settings(port),
        create_process_owner(),
    )
    await server.start()
    assert server.is_running
    await server.stop()
    await server.stop()
    assert not server.is_running


def test_managed_server_rejects_occupied_port_before_spawn():
    asyncio.run(_test_managed_server_rejects_occupied_port_before_spawn())


async def _test_managed_server_rejects_occupied_port_before_spawn():
    port = _free_port()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
        listener.listen()
        server = ManagedLlamaServer(
            [sys.executable, '-c', 'raise SystemExit(1)'],
            f'http://127.0.0.1:{port}',
            '/health',
            _server_settings(port),
            create_process_owner(),
        )
        with pytest.raises(LocalModelPortInUse):
            await server.start()
        assert not server.is_running


def test_managed_server_reports_child_exit():
    asyncio.run(_test_managed_server_reports_child_exit())


async def _test_managed_server_reports_child_exit():
    port = _free_port()
    server = ManagedLlamaServer(
        [sys.executable, '-c', 'raise SystemExit(7)'],
        f'http://127.0.0.1:{port}',
        '/health',
        _server_settings(port),
        create_process_owner(),
    )
    with pytest.raises(LocalModelStartError, match='code 7'):
        await server.start()
    assert not server.is_running


def test_managed_server_kills_unresponsive_child_after_timeout():
    asyncio.run(_test_managed_server_kills_unresponsive_child_after_timeout())


async def _test_managed_server_kills_unresponsive_child_after_timeout():
    port = _free_port()
    settings = _server_settings(port).model_copy(update={'ready_timeout': 0.1})
    server = ManagedLlamaServer(
        [sys.executable, '-c', 'import time; time.sleep(60)'],
        f'http://127.0.0.1:{port}',
        '/health',
        settings,
        create_process_owner(),
    )
    with pytest.raises(TimeoutError):
        await server.start()
    assert not server.is_running


@pytest.mark.skipif(
    sys.platform != 'linux', reason='Linux parent-death contract'
)
def test_linux_child_exits_when_parent_dies(tmp_path):
    child_pid_file = tmp_path / 'child.pid'
    parent_script = """
import os
import pathlib
import subprocess
import sys
import time
path = pathlib.Path(sys.argv[1])
environment = os.environ.copy()
environment['KMS2_PARENT_PID'] = str(os.getpid())
child = subprocess.Popen([
    sys.executable,
    '-m',
    'kms2.local_models.linux_child',
    '--',
    sys.executable,
    '-c',
    'import time; time.sleep(60)',
], env=environment)
path.write_text(str(child.pid))
time.sleep(60)
"""
    parent = subprocess.Popen(
        [sys.executable, '-c', parent_script, str(child_pid_file)]
    )
    deadline = time.monotonic() + 2
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_file.read_text())
    parent.terminate()
    parent.wait(timeout=2)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f'/proc/{child_pid}/stat')
        if not stat_path.exists():
            break
        if stat_path.read_text().split()[2] == 'Z':
            break
        time.sleep(0.01)
    else:
        os.kill(child_pid, 9)
        pytest.fail('child survived parent death')


def test_posix_child_executes_supplied_command():
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'kms2.local_models.posix_child',
            '--',
            sys.executable,
            '-c',
            'raise SystemExit(0)',
        ],
        check=False,
    )
    assert result.returncode == 0


def test_posix_owner_selects_platform_bootstrap(monkeypatch):
    monkeypatch.setattr(posix_ownership.sys, 'platform', 'linux')
    assert posix_ownership._posix_bootstrap_module() == (
        'kms2.local_models.linux_child'
    )
    monkeypatch.setattr(posix_ownership.sys, 'platform', 'darwin')
    assert posix_ownership._posix_bootstrap_module() == (
        'kms2.local_models.posix_child'
    )


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process-group contract')
def test_posix_process_owner_force_kills_and_reaps_child(tmp_path):
    asyncio.run(_test_posix_process_owner_force_kills_and_reaps_child(tmp_path))


async def _test_posix_process_owner_force_kills_and_reaps_child(tmp_path):
    pid_path = tmp_path / 'child.pid'
    owner = PosixProcessOwner()
    process = await owner.spawn(
        [
            sys.executable,
            '-c',
            (
                'import pathlib, signal, sys, time; '
                'pathlib.Path(sys.argv[1]).write_text(str(__import__("os").getpid())); '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                'time.sleep(60)'
            ),
            str(pid_path),
        ]
    )
    deadline = time.monotonic() + 2
    while not pid_path.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    child_pid = int(pid_path.read_text())

    await process.stop(0.1)
    await process.stop(0.1)

    assert process.returncode is None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f'/proc/{child_pid}/stat')
        if not stat_path.exists() or stat_path.read_text().split()[2] == 'Z':
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail('owned child remained running after stop')


def test_process_owner_factory_selects_posix_backend(monkeypatch):
    assert isinstance(create_process_owner(), PosixProcessOwner)
    monkeypatch.setattr(ownership.os, 'name', 'nt')
    with pytest.raises(
        RuntimeError,
        match='local-model process ownership is not implemented on this platform',
    ):
        create_process_owner()


def test_runtime_exports_single_public_resource():
    assert LocalModelRuntime.__name__ == 'LocalModelRuntime'


def test_runtime_serializes_role_transitions_and_cleans_failed_activation(
    monkeypatch,
):
    asyncio.run(
        _test_runtime_serializes_role_transitions_and_cleans_failed_activation(
            monkeypatch
        )
    )


async def _test_runtime_serializes_role_transitions_and_cleans_failed_activation(
    monkeypatch,
):
    activated: list[tuple[ResidentRole, str | None]] = []
    released: list[tuple[ResidentRole, str | None] | None] = []

    async def activate(self, role, profile):
        activated.append((role, profile))

    async def release(self):
        released.append(self._active_role)

    monkeypatch.setattr(LocalModelRuntime, '_activate', activate)
    monkeypatch.setattr(LocalModelRuntime, '_release_active_resources', release)
    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
    runtime._state = 'started'

    async def operation():
        return 'ok'

    assert await runtime._execute(ResidentRole.LLM, 'text', operation) == 'ok'
    assert await runtime._execute(ResidentRole.LLM, 'text', operation) == 'ok'
    await runtime._execute(ResidentRole.EMBEDDING, None, operation)
    await runtime._execute(ResidentRole.RERANKER, None, operation)
    await runtime._execute(ResidentRole.LLM, 'vision', operation)

    assert activated == [
        (ResidentRole.LLM, 'text'),
        (ResidentRole.EMBEDDING, None),
        (ResidentRole.RERANKER, None),
        (ResidentRole.LLM, 'vision'),
    ]
    assert released == [
        (ResidentRole.LLM, 'text'),
        (ResidentRole.EMBEDDING, None),
        (ResidentRole.RERANKER, None),
    ]

    async def fail_activate(self, role, profile):
        raise asyncio.CancelledError

    monkeypatch.setattr(LocalModelRuntime, '_activate', fail_activate)
    with pytest.raises(asyncio.CancelledError):
        await runtime._execute(ResidentRole.EMBEDDING, None, operation)
    assert runtime._active_role is None
    assert released[-1] == (ResidentRole.EMBEDDING, None)
    await runtime.close()


def test_runtime_executes_embedding_and_reranking_roles():
    asyncio.run(_test_runtime_executes_embedding_and_reranking_roles())


async def _test_runtime_executes_embedding_and_reranking_roles():
    settings = LocalModelRuntimeSettings()
    runtime = LocalModelRuntime(settings)
    runtime._state = 'started'
    requests: list[tuple[str, dict[str, object]]] = []

    class FakeServer:
        def __init__(self):
            self.started = 0
            self.stopped = 0

        async def start(self):
            self.started += 1

        async def stop(self):
            self.stopped += 1

    embedding_server = FakeServer()
    reranker_server = FakeServer()

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.read()).json()
        requests.append((request.url.path, body))
        if request.url.path == '/v1/embeddings':
            return httpx.Response(
                200,
                json={
                    'data': [
                        {'embedding': [1.0, 2.0]},
                        {'embedding': [3.0, 4.0]},
                    ]
                },
            )
        return httpx.Response(
            200,
            json={'results': [{'index': 0, 'relevance_score': 0.75}]},
        )

    runtime._embedding_server = embedding_server
    runtime._reranker_server = reranker_server
    runtime._embedding_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url='http://embedding',
    )
    runtime._reranker_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url='http://reranker',
    )
    try:
        assert await runtime.embed(['a', 'b']) == [[1.0, 2.0], [3.0, 4.0]]
        assert await runtime.rerank('query', ['a', 'b']) == [
            {'index': 0, 'relevance_score': 0.75}
        ]
        assert embedding_server.started == 1
        assert embedding_server.stopped == 1
        assert reranker_server.started == 1
        assert requests == [
            (
                '/v1/embeddings',
                {'model': 'Qwen3-Embedding-8B', 'input': ['a', 'b']},
            ),
            (
                '/v1/rerank',
                {
                    'model': 'Qwen3-Reranker-8B',
                    'query': 'query',
                    'documents': ['a', 'b'],
                },
            ),
        ]
    finally:
        await runtime.close()


def test_retrieval_payload_contracts_are_preserved():
    requests: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.read()).json()
        requests.append((request.url.path, body))
        if request.url.path == '/v1/embeddings':
            return httpx.Response(
                200,
                json={
                    'data': [
                        {'index': 1, 'embedding': [0.0, 4.0]},
                        {'index': 0, 'embedding': [3.0, 4.0]},
                    ]
                },
            )
        return httpx.Response(200, json={'results': [{'index': 1}]})

    async def exercise():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url='http://local',
        )
        embedding_response = await client.post(
            '/v1/embeddings',
            json={'model': 'Qwen3-Embedding-8B', 'input': ['a', 'b']},
        )
        rerank_response = await client.post(
            '/v1/rerank',
            json={
                'model': 'Qwen3-Reranker-8B',
                'query': 'query',
                'documents': ['a', 'b'],
                'top_n': 1,
            },
        )
        await client.aclose()
        return embedding_response.json(), rerank_response.json()

    embedding, rerank = asyncio.run(exercise())
    assert embedding['data'][0]['embedding'] == [0.0, 4.0]
    assert rerank['results'] == [{'index': 1}]
    assert requests[0][0] == '/v1/embeddings'
    assert requests[1][0] == '/v1/rerank'
