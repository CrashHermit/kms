"""Tests for invocation-scoped external research resources."""

import asyncio

import pytest

from kms.composition import research as research_composition
from kms.config.research import ResearchSettings, TavilySettings
from kms.config.settings import Settings


class _TrackedClient:
    """Record whether a partially acquired async client is closed."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        """Record client shutdown."""
        self.closed = True


def test_research_resources_close_partial_client_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close acquired clients when later setup fails."""

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

        with pytest.raises(RuntimeError, match='media client setup failed'):
            async with research_composition.research_resources(settings):
                pytest.fail('resource setup should fail before yielding')

        assert clients[0].closed is True

    asyncio.run(exercise())


def test_gotenberg_url_defaults_to_loopback() -> None:
    """Keep the renderer endpoint on the local host by default."""
    assert ResearchSettings().gotenberg_url == 'http://127.0.0.1:3000'
