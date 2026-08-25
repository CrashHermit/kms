"""Own the KMS process lifecycle and shared infrastructure."""

from collections.abc import Callable
from pathlib import Path

from kms import config
from kms.core import serve
from kms.graph import db


class Runtime:
    """Own process-wide KMS services for one or more ingestion runs."""

    def __init__(self) -> None:
        self._neo4j_configured = db.is_configured()
        self._model_manager = (
            serve.RouterManager(serve.default_router())
            if config.get_settings().serving.manage
            else None
        )

    @property
    def retrieval_server_manager(self) -> serve.RetrievalServerManager | None:
        retrieval = getattr(config.get_settings().serving, 'retrieval', None)
        if retrieval is None:
            return None
        if retrieval.embedding.manage or retrieval.reranker.manage:
            return serve.retrieval_server_manager()
        return None


    @property
    def model_manager(self) -> serve.RouterManager | None:
        """Return the shared local model manager, if enabled."""
        return self._model_manager

    @property
    def neo4j_configured(self) -> bool:
        """Return whether Neo4j is configured for this process."""
        return self._neo4j_configured

    def session_factory(self) -> Callable | None:
        """Return the shared Neo4j session factory, if configured."""
        return db.session if self._neo4j_configured else None
    async def __aenter__(self) -> 'Runtime':
        """Start the runtime context."""
        return self

    async def close(self) -> None:
        """Stop shared services owned by this runtime."""
        from kms.core import embeddings, reranker

        await embeddings.close_retrieval_clients()
        await reranker.close_retrieval_clients()
        if self.retrieval_server_manager:
            self.retrieval_server_manager.shutdown()
        if self._model_manager:
            self._model_manager.shutdown()
        await db.close_driver()
    async def __aexit__(self, exception_type, exception, traceback) -> None:
        """Stop shared services after the process context exits."""
        await self.close()


    async def ingest(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = 'output',
        pages: list[int] | None = None,
        source: str | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> dict:
        """Ingest one document using this runtime's shared services."""
        from kms.construction import runner

        return await runner.ingest(
            self,
            pdf_path,
            output_dir=output_dir,
            pages=pages,
            source=source,
            title=title,
            author=author,
        )


async def ingest(
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    pages: list[int] | None = None,
    source: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> dict:
    """Ingest one document in a short-lived runtime context."""
    async with Runtime() as application:
        return await application.ingest(
            pdf_path,
            output_dir=output_dir,
            pages=pages,
            source=source,
            title=title,
            author=author,
        )
