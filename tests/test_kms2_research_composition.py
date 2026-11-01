"""Tests for invocation-scoped external research resources."""

import asyncio
from pathlib import Path

import pytest

from kms2.composition import research as research_composition
from kms2.config.research import ResearchSettings, TavilySettings
from kms2.config.settings import Settings


class _TrackedClient:
    """Record whether a partially acquired async client is closed."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        """Record client shutdown."""
        self.closed = True


def test_research_resources_close_partial_client_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close acquired clients and temporary files if later setup fails."""

    async def exercise() -> None:
        settings = Settings(
            _env_file=None,
            research=ResearchSettings(
                enabled=True,
                tavily=TavilySettings(api_key='test-key'),
            ),
        )
        clients: list[_TrackedClient] = []

        def create_client(*args: object, **kwargs: object) -> _TrackedClient:
            if clients:
                raise RuntimeError('media client setup failed')
            client = _TrackedClient()
            clients.append(client)
            return client

        monkeypatch.setattr(
            research_composition.httpx,
            'AsyncClient',
            create_client,
        )
        monkeypatch.setattr(
            research_composition.tempfile,
            'tempdir',
            str(tmp_path),
        )

        with pytest.raises(RuntimeError, match='media client setup failed'):
            async with research_composition.research_resources(settings):
                pytest.fail('resource setup should fail before yielding')

        assert clients[0].closed is True
        assert list(tmp_path.iterdir()) == []

    asyncio.run(exercise())
