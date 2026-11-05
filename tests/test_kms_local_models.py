import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kms.config.runtime import LlamaServerSettings, LocalModelRuntimeSettings
from kms.local_models import (
    ownership,
    posix_ownership,
)
from kms.local_models.ownership import create_process_owner
from kms.local_models.posix_ownership import PosixProcessOwner
from kms.local_models.process import (
    LocalModelPortInUse,
    LocalModelStartError,
    ManagedLlamaServer,
)
from kms.local_models.router import (
    _model_server_presets_ini,
    _router_command,
)
from kms.local_models.runtime import LocalModelRuntime


def test_router_command_has_isolated_api_prefix():
    router = _router_command(
        LocalModelRuntimeSettings().router,
        Path('/tmp/models.ini'),
        '/kms-test',
    )

    assert router[-5:] == [
        '--models-preset',
        '/tmp/models.ini',
        '--models-max',
        '1',
        '--no-models-autoload',
    ]


def test_router_preset_contains_text_and_vision_model_server_profiles():
    preset = _model_server_presets_ini(LocalModelRuntimeSettings().router)

    assert '[gemma-text-32k]' in preset
    assert '[gemma-vision-8k]' in preset
    assert '[gemma-4-e2b-text]' in preset
    assert 'reasoning = off' in preset
    assert 'ctx-size = 32768' in preset
    assert preset.count('reasoning = on') == 3
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
environment['KMS_PARENT_PID'] = str(os.getpid())
child = subprocess.Popen([
    sys.executable,
    '-m',
    'kms.local_models.linux_child',
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
            'kms.local_models.posix_child',
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
        'kms.local_models.linux_child'
    )
    monkeypatch.setattr(posix_ownership.sys, 'platform', 'darwin')
    assert posix_ownership._posix_bootstrap_module() == (
        'kms.local_models.posix_child'
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


def test_runtime_defers_router_start_until_local_llm_use(monkeypatch):
    asyncio.run(
        _test_runtime_defers_router_start_until_local_llm_use(monkeypatch)
    )


async def _test_runtime_defers_router_start_until_local_llm_use(monkeypatch):
    routers = []

    class FakeRouter:
        endpoint = 'http://router'

        def __init__(self, *args):
            self.start_count = 0
            routers.append(self)

        async def start(self):
            self.start_count += 1

        async def ensure_model_server(self, profile):
            self.profile = profile

        async def release_model_server(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr('kms.local_models.runtime.RouterServer', FakeRouter)

    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
    await runtime.start()
    assert routers[0].start_count == 0
    await runtime.close()
    assert routers[0].start_count == 0

    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
    await runtime.start()

    async def operation():
        return 'local'

    assert await runtime._execute('gemma-text-32k', operation) == 'local'
    assert routers[1].start_count == 1
    assert routers[1].profile == 'gemma-text-32k'
    await runtime.close()


def test_runtime_serializes_profile_transitions_and_cleans_failed_activation(
    monkeypatch,
):
    asyncio.run(
        _test_runtime_serializes_profile_transitions_and_cleans_failed_activation(
            monkeypatch
        )
    )


async def _test_runtime_serializes_profile_transitions_and_cleans_failed_activation(
    monkeypatch,
):
    activated: list[str] = []
    released: list[str | None] = []

    async def activate(self, profile):
        activated.append(profile)

    async def release(self):
        released.append(self._active_profile)

    monkeypatch.setattr(LocalModelRuntime, '_activate', activate)
    monkeypatch.setattr(LocalModelRuntime, '_release_active_resources', release)
    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
    runtime._state = 'started'

    async def operation():
        return 'ok'

    assert await runtime._execute('text', operation) == 'ok'
    assert await runtime._execute('text', operation) == 'ok'
    assert await runtime._execute('vision', operation) == 'ok'
    assert await runtime._execute('text', operation) == 'ok'

    assert activated == ['text', 'vision', 'text']
    assert released == ['text', 'vision']

    async def fail_activate(self, profile):
        raise asyncio.CancelledError

    monkeypatch.setattr(LocalModelRuntime, '_activate', fail_activate)
    with pytest.raises(asyncio.CancelledError):
        await runtime._execute('failed', operation)
    assert runtime._active_profile is None
    assert released[-2:] == ['text', 'failed']
    await runtime.close()
