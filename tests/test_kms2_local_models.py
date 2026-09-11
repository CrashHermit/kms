import asyncio
from pathlib import Path

import httpx
import pytest

from kms2.config import LocalModelRuntimeSettings
from kms2.local_models.coordinator import (
    _GpuCoordinator,
    _GpuResidency,
    _GpuRole,
)
from kms2.local_models.dedicated import _dedicated_command
from kms2.local_models.inference import _EmbeddingClient, _RerankerClient
from kms2.local_models.router import _model_server_presets_ini, _router_command


def test_local_server_commands_have_isolated_api_prefixes():
    settings = LocalModelRuntimeSettings()

    router = _router_command(
        settings.router, Path('/tmp/models.ini'), '/kms2-test'
    )
    embedding = _dedicated_command(
        settings.embedding,
        '/kms2-test',
        reranking=False,
    )
    reranker = _dedicated_command(
        settings.reranker,
        '/kms2-test',
        reranking=True,
    )

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


def test_router_preset_contains_both_model_server_profiles():
    preset = _model_server_presets_ini(LocalModelRuntimeSettings().router)

    assert '[gemma-text-32k]' in preset
    assert '[gemma-vision-8k]' in preset
    assert 'ctx-size = 32768' in preset
    assert 'no-warmup = true' in preset
    assert 'mmproj = ' in preset


def test_gpu_coordinator_serializes_roles_and_releases_resident_role():
    asyncio.run(
        _test_gpu_coordinator_serializes_roles_and_releases_resident_role()
    )


async def _test_gpu_coordinator_serializes_roles_and_releases_resident_role():
    coordinator = _GpuCoordinator()
    events: list[str] = []

    async def activate(role: str) -> None:
        events.append(f'activate:{role}')

    async def deactivate(role: str) -> None:
        events.append(f'deactivate:{role}')

    async def operation(role: str) -> str:
        events.append(f'operation:{role}')
        return role

    assert (
        await coordinator.execute(
            _GpuResidency(_GpuRole.EMBEDDING),
            lambda: activate('embedding'),
            lambda: deactivate('embedding'),
            lambda: operation('embedding'),
        )
        == 'embedding'
    )
    assert (
        await coordinator.execute(
            _GpuResidency(_GpuRole.RERANKER),
            lambda: activate('reranker'),
            lambda: deactivate('reranker'),
            lambda: operation('reranker'),
        )
        == 'reranker'
    )
    await coordinator.close()

    assert events == [
        'activate:embedding',
        'operation:embedding',
        'deactivate:embedding',
        'activate:reranker',
        'operation:reranker',
        'deactivate:reranker',
    ]


def test_gpu_coordinator_rejects_queued_work_after_shutdown_starts():
    asyncio.run(
        _test_gpu_coordinator_rejects_queued_work_after_shutdown_starts()
    )


async def _test_gpu_coordinator_rejects_queued_work_after_shutdown_starts():
    coordinator = _GpuCoordinator()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    events: list[str] = []

    async def activate(role: str) -> None:
        events.append(f'activate:{role}')

    async def deactivate(role: str) -> None:
        events.append(f'deactivate:{role}')

    async def first_operation() -> str:
        events.append('operation:embedding')
        first_started.set()
        await release_first.wait()
        return 'embedding'

    async def queued_operation() -> str:
        events.append('operation:reranker')
        return 'reranker'

    first_task = asyncio.create_task(
        coordinator.execute(
            _GpuResidency(_GpuRole.EMBEDDING),
            lambda: activate('embedding'),
            lambda: deactivate('embedding'),
            first_operation,
        )
    )
    await first_started.wait()
    queued_task = asyncio.create_task(
        coordinator.execute(
            _GpuResidency(_GpuRole.RERANKER),
            lambda: activate('reranker'),
            lambda: deactivate('reranker'),
            queued_operation,
        )
    )
    close_task = asyncio.create_task(coordinator.close())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert coordinator._closed is True
    release_first.set()

    assert await first_task == 'embedding'
    await close_task
    with pytest.raises(RuntimeError, match='local model runtime is closed'):
        await queued_task

    assert events == [
        'activate:embedding',
        'operation:embedding',
        'deactivate:embedding',
    ]


def test_retrieval_clients_preserve_server_payload_order():
    asyncio.run(_test_retrieval_clients_preserve_server_payload_order())


async def _test_retrieval_clients_preserve_server_payload_order():
    requests: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.read()).json()
        requests.append((request.url.path, body))
        if request.url.path == '/v1/embeddings':
            return httpx.Response(
                200,
                json={
                    'data': [
                        {'index': 1, 'embedding': [1.0]},
                        {'index': 0, 'embedding': [0.0]},
                    ]
                },
            )
        return httpx.Response(200, json={'results': [{'index': 1}]})

    class Role:
        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    settings = LocalModelRuntimeSettings()
    coordinator = _GpuCoordinator()
    role = Role()
    embedding_http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url='http://embedding',
    )
    reranker_http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url='http://reranker',
    )
    embedding = _EmbeddingClient(
        settings.embedding, embedding_http, coordinator, role
    )
    reranker = _RerankerClient(
        settings.reranker, reranker_http, coordinator, role
    )

    assert await embedding.embed(['a', 'b']) == [[1.0], [0.0]]
    assert await reranker.rerank('query', ['a', 'b'], top_n=1) == [{'index': 1}]

    await coordinator.close()
    await embedding_http.aclose()
    await reranker_http.aclose()
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
                'top_n': 1,
            },
        ),
    ]


def test_runtime_eagerly_starts_only_its_empty_router(monkeypatch, tmp_path):
    asyncio.run(
        _test_runtime_eagerly_starts_only_its_empty_router(
            monkeypatch, tmp_path
        )
    )


async def _test_runtime_eagerly_starts_only_its_empty_router(
    monkeypatch,
    tmp_path: Path,
):
    events: list[str] = []

    class Router:
        endpoint = 'http://router/kms2-test'

        def __init__(self, *args: object) -> None:
            del args

        async def start(self) -> None:
            events.append('router:start')

        async def close(self) -> None:
            events.append('router:close')

    class Role:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.endpoint = 'http://retrieval/kms2-test'

        async def close(self) -> None:
            events.append('role:close')

    class Client:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def aclose(self) -> None:
            events.append('client:close')

    directory = type(
        'Directory',
        (),
        {
            'name': str(tmp_path),
            'cleanup': lambda self: events.append('cleanup'),
        },
    )
    monkeypatch.setattr(
        'kms2.local_models.runtime.tempfile.TemporaryDirectory',
        lambda **kwargs: directory(),
    )
    monkeypatch.setattr('kms2.local_models.runtime._RouterServer', Router)
    monkeypatch.setattr('kms2.local_models.runtime._DedicatedRole', Role)
    monkeypatch.setattr('kms2.local_models.runtime.httpx.AsyncClient', Client)

    from kms2.local_models import LocalModelRuntime

    runtime = LocalModelRuntime(LocalModelRuntimeSettings())
    await runtime.start()

    assert events == ['router:start']
    assert runtime.embedding is not None
    assert runtime.reranker is not None

    await runtime.close()

    assert events == [
        'router:start',
        'client:close',
        'client:close',
        'role:close',
        'role:close',
        'router:close',
        'cleanup',
    ]
