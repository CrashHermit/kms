"""Application entry points for running KMS pipelines."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from kms.config.inference import StageInferenceSettings
from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.chatgpt import (
    ensure_chatgpt_authenticated,
)
from kms.inference.chatgpt import (
    login_chatgpt as _login_chatgpt,
)
from kms.inference.embedding.resources import (
    EmbeddingResources,
    embedding_resources,
)
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.local_models.runtime import LocalModelRuntime
from kms.source.capture.models import PDFCandidate
from kms.source.capture.resources import capture_resources
from kms.source.composition import build_source_ingestion_service
from kms.source.discovery.composition import build_source_discovery
from kms.source.discovery.models import SourceCandidate
from kms.source.discovery.resources import discovery_resources
from kms.source.models.inputs import CapturedPDFSource
from kms.source.models.snapshot import SourceSnapshot
from kms.source.models.source import Source
from kms.source.retrieval.composition import (
    build_search_selection_service,
    build_source_reader,
    build_source_search_service,
)
from kms.source.retrieval.models import (
    SourceImage,
    SourcePassage,
    SourceProcedurePassage,
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementPassage,
    SourceStatementSearchHit,
)
from kms.source.storage.catalog import SourceCatalogRepository
from kms.train.recorder import Recorder
from kms.users.models import User
from kms.users.repository import UserRepository
from kms.users.service import UserService


@dataclass(frozen=True, slots=True)
class _ApplicationResources:
    """Resources shared by the stages of one application invocation."""

    local_models: LocalModelRuntime
    embeddings: EmbeddingResources | None
    tokenizers: LanguageModelTokenizers
    database: DatabaseClient
    recorder: Recorder | None


@asynccontextmanager
async def _application_resources(
    settings: Settings,
    *,
    embeddings_required: bool = True,
) -> AsyncIterator[_ApplicationResources]:
    """Own invocation resources and acquire retrieval only when requested."""
    database = DatabaseClient(settings.database)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = LanguageModelTokenizers(settings)
    try:
        async with AsyncExitStack() as stack:
            local_models = await stack.enter_async_context(
                LocalModelRuntime(settings.local_models)
            )
            embeddings = (
                await stack.enter_async_context(embedding_resources(settings))
                if embeddings_required
                else None
            )
            if embeddings is None:
                await schema.ensure_structural_schema(database.session)
            else:
                await schema.ensure_schema(
                    database.session,
                    embedding_dimension=embeddings.model.dimension,
                )
            yield _ApplicationResources(
                local_models=local_models,
                embeddings=embeddings,
                tokenizers=tokenizers,
                database=database,
                recorder=recorder,
            )
    finally:
        await database.close()


def login_chatgpt() -> None:
    """Start LiteLLM's explicit ChatGPT device login from the terminal."""
    _login_chatgpt()


async def _preflight_chatgpt(
    settings: Settings,
    *stage_settings: BaseModel,
) -> None:
    """Require existing ChatGPT credentials for selected inference stages."""
    for stage in stage_settings:
        if _contains_chatgpt_inference(stage, settings):
            await ensure_chatgpt_authenticated()
            return


def _contains_chatgpt_inference(
    value: object,
    settings: Settings,
) -> bool:
    """Resolve inference leaves within one explicitly relevant stage tree."""
    if isinstance(value, StageInferenceSettings):
        return settings.llm.resolve(value).backend == 'chatgpt'
    if isinstance(value, BaseModel):
        return any(
            _contains_chatgpt_inference(field, settings)
            for field in value.__dict__.values()
        )
    if isinstance(value, dict):
        return any(
            _contains_chatgpt_inference(field, settings)
            for field in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_chatgpt_inference(field, settings) for field in value
        )
    return False


@asynccontextmanager
async def _database_resources(
    settings: Settings,
) -> AsyncIterator[DatabaseClient]:
    """Own database resources for commands that do not need local models."""
    database = DatabaseClient(settings.database)
    try:
        await schema.ensure_structural_schema(database.session)
        yield database
    finally:
        await database.close()


async def list_users(settings: Settings) -> list[User]:
    """Return users for terminal onboarding and selection."""
    database = DatabaseClient(settings.database)
    try:
        return await UserService(UserRepository(database.session)).list_users()
    finally:
        await database.close()


async def create_user(settings: Settings, name: str) -> User:
    """Create a user after ensuring only structural schema."""
    database = DatabaseClient(settings.database)
    try:
        await schema.ensure_structural_schema(database.session)
        return await UserService(UserRepository(database.session)).create_user(
            name
        )
    finally:
        await database.close()


async def list_unowned_sources(settings: Settings) -> list[Source]:
    """Return legacy sources available for explicit adoption."""
    database = DatabaseClient(settings.database)
    try:
        return await SourceCatalogRepository(
            database.session
        ).list_unowned_sources()
    finally:
        await database.close()


async def adopt_source(
    settings: Settings,
    user_uuid: str,
    source_uuid: str,
) -> None:
    """Explicitly associate an unowned source with a user."""
    database = DatabaseClient(settings.database)
    try:
        await SourceCatalogRepository(database.session).adopt_source(
            user_uuid,
            source_uuid,
        )
    finally:
        await database.close()


async def list_sources(settings: Settings, user_uuid: str) -> list[Source]:
    """Return persisted sources owned by one user."""
    database = DatabaseClient(settings.database)
    try:
        return await SourceCatalogRepository(database.session).list_sources(
            user_uuid
        )
    finally:
        await database.close()


async def search_sources(
    settings: Settings,
    user_uuid: str,
    query: str,
    *,
    source_uuids: list[str] | None = None,
) -> list[SourceSearchHit]:
    """Search a user's source blocks with the configured embedding."""

    if source_uuids == []:
        return []
    await _preflight_chatgpt(settings, settings.search_selection)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        selection = build_search_selection_service(
            settings, predictors=predictors, tokenizers=resources.tokenizers
        )
        service = build_source_search_service(
            resources.database,
            embedder=resources.embeddings.embedder,
            settings=settings.source_search,
            selection=selection,
        )
        return await service.search_sources(
            user_uuid, query, source_uuids=source_uuids
        )


async def read_source_passages(
    settings: Settings,
    user_uuid: str,
    block_uuids: list[str],
    *,
    expand_memberships: bool = True,
    before: int = 0,
    after: int = 0,
) -> list[SourcePassage]:
    """Read authorized source passages and their direct evidence groups."""
    async with _database_resources(settings) as database:
        service = build_source_reader(database)
        return await service.read_passages(
            user_uuid,
            block_uuids,
            expand_memberships=expand_memberships,
            before=before,
            after=after,
        )


async def read_source_images(
    settings: Settings,
    user_uuid: str,
    block_uuids: list[str],
) -> list[SourceImage]:
    """Read verified image bytes owned by the requested source blocks."""
    async with _database_resources(settings) as database:
        service = build_source_reader(database)
        return await service.read_images(user_uuid, block_uuids)


async def search_source_statements(
    settings: Settings,
    user_uuid: str,
    query: str,
    *,
    source_uuids: list[str] | None = None,
    is_exercise: bool | None = None,
) -> list[SourceStatementSearchHit]:
    """Search reconstructed statements in the user's eligible sources."""
    if source_uuids == []:
        return []
    await _preflight_chatgpt(settings, settings.search_selection)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        selection = build_search_selection_service(
            settings, predictors=predictors, tokenizers=resources.tokenizers
        )
        service = build_source_search_service(
            resources.database,
            embedder=resources.embeddings.embedder,
            settings=settings.source_search,
            selection=selection,
        )
        return await service.search_statements(
            user_uuid,
            query,
            source_uuids=source_uuids,
            is_exercise=is_exercise,
        )


async def search_source_procedures(
    settings: Settings,
    user_uuid: str,
    query: str,
    *,
    source_uuids: list[str] | None = None,
) -> list[SourceProcedureSearchHit]:
    """Search reconstructed procedures without mixing material-type scores."""
    if source_uuids == []:
        return []
    await _preflight_chatgpt(settings, settings.search_selection)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        selection = build_search_selection_service(
            settings, predictors=predictors, tokenizers=resources.tokenizers
        )
        service = build_source_search_service(
            resources.database,
            embedder=resources.embeddings.embedder,
            settings=settings.source_search,
            selection=selection,
        )
        return await service.search_procedures(
            user_uuid, query, source_uuids=source_uuids
        )


async def read_source_statements(
    settings: Settings, user_uuid: str, statement_uuids: list[str]
) -> list[SourceStatementPassage]:
    """Read statements with their original member evidence and instructions."""
    async with _database_resources(settings) as database:
        return await build_source_reader(database).read_statements(
            user_uuid, statement_uuids
        )


async def read_source_procedures(
    settings: Settings, user_uuid: str, procedure_uuids: list[str]
) -> list[SourceProcedurePassage]:
    """Read procedures with their ordinary source passages."""
    async with _database_resources(settings) as database:
        return await build_source_reader(database).read_procedures(
            user_uuid, procedure_uuids
        )


async def ingest_source(
    settings: Settings,
    user_uuid: str,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceSnapshot:
    """Ingest one local PDF as a complete immutable owned source."""
    data = await asyncio.to_thread(Path(pdf_path).read_bytes)
    capture = CapturedPDFSource(
        data=data,
        title=Path(pdf_path).name,
        url=None,
        retrieved_at=datetime.now(UTC),
    )
    return await ingest_captured_source(
        settings, user_uuid, capture, pages=pages
    )


async def ingest_captured_source(
    settings: Settings,
    user_uuid: str,
    capture: CapturedPDFSource,
    *,
    pages: list[int] | None = None,
) -> SourceSnapshot:
    """Process and publish a captured PDF as an owned source."""
    async with _database_resources(settings) as database:
        await UserRepository(database.session).read_user(user_uuid)
    await _preflight_chatgpt(settings, settings.source_processing)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        ingestion = build_source_ingestion_service(
            settings,
            resources.database,
            predictors=predictors,
            embeddings=resources.embeddings,
            tokenizers=resources.tokenizers,
        )
        return await ingestion.ingest(user_uuid, capture, pages=pages)


async def discover_sources(
    settings: Settings, question: str
) -> list[SourceCandidate]:
    """Search and select useful links without capturing or ingesting them."""
    await _preflight_chatgpt(settings, settings.source_discovery.inference)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    async with AsyncExitStack() as stack:
        local_models = await stack.enter_async_context(
            LocalModelRuntime(settings.local_models)
        )
        web = await stack.enter_async_context(
            discovery_resources(settings.source_discovery)
        )
        predictors = PredictorFactory(
            settings,
            local_models,
            recorder,
            tokenizers=LanguageModelTokenizers(settings),
        )
        discovery = build_source_discovery(
            settings, web=web, predictors=predictors
        )
        return await discovery.discover(question)


async def capture_source(
    settings: Settings, url: str, *, title: str | None = None
) -> CapturedPDFSource | PDFCandidate:
    """Capture a web page as a PDF, or return a PDF link for manual handling."""
    async with capture_resources(settings.source_capture) as capture:
        return await capture.capture(url, title=title)
