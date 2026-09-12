"""Application entry points for running KMS2 pipelines."""

from dataclasses import dataclass
from pathlib import Path

from kms2.composition import build_semantic_graph, build_source_graph
from kms2.config import Settings
from kms2.core.model.source import Source
from kms2.database.client import DatabaseClient
from kms2.database.source.repository import SourceRepository
from kms2.local_models import LocalModelRuntime


@dataclass(frozen=True, slots=True)
class SourceIngestionResult:
    """Summary of one completed source ingestion."""

    source: Source
    page_count: int


@dataclass(frozen=True, slots=True)
class SourceProcessingResult:
    """Summary of one completed source ingestion and raw triplet extraction."""

    source: Source
    page_count: int
    triplet_count: int


@dataclass(frozen=True, slots=True)
class SemanticStageResult:
    """Counts persisted by one complete semantic stage."""

    raw_assertion_count: int
    entity_enrichment_count: int
    event_enrichment_count: int
    predicate_enrichment_count: int


@dataclass(frozen=True, slots=True)
class SourceSemanticStageResult:
    """Summary of source ingestion and its complete semantic stage."""

    source: Source
    page_count: int
    semantic: SemanticStageResult


async def list_sources(settings: Settings) -> list[Source]:
    """Return persisted sources for interactive source selection."""
    database = DatabaseClient(settings.database)
    try:
        return await SourceRepository(database.session).list_sources()
    finally:
        await database.close()


async def _ingest_source_with_resources(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> SourceIngestionResult:
    source = Source(key=Path(pdf_path).name)
    initial_state = {
        'pdf_path': pdf_path,
        'pages': pages,
        'source': source,
    }
    graph = build_source_graph(settings, local_models, database).build_graph()
    final_state = await graph.ainvoke(initial_state)
    return SourceIngestionResult(
        source=final_state['source'],
        page_count=len(final_state['split_pages']),
    )


async def _extract_triplets_with_resources(
    settings: Settings,
    source_uuid: str,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> int:
    graph = build_semantic_graph(settings, local_models, database).build_graph(
        raw_only=True
    )
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return len(final_state['raw_assertions'])


async def _run_semantic_stage_with_resources(
    settings: Settings,
    source_uuid: str,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> SemanticStageResult:
    """Run one complete semantic graph in dependency order."""
    graph = build_semantic_graph(settings, local_models, database).build_graph()
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return SemanticStageResult(
        raw_assertion_count=len(final_state['raw_assertions']),
        entity_enrichment_count=final_state['entity_persisted_count'],
        event_enrichment_count=final_state['event_persisted_count'],
        predicate_enrichment_count=final_state['predicate_persisted_count'],
    )


async def ingest_source(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceIngestionResult:
    """Run the source pipeline and return its persisted-source summary."""
    database = DatabaseClient(settings.database)

    try:
        async with LocalModelRuntime(settings.local_models) as local_models:
            return await _ingest_source_with_resources(
                settings,
                pdf_path,
                pages,
                local_models,
                database,
            )
    finally:
        await database.close()


async def extract_triplets(settings: Settings, source_uuid: str) -> int:
    """Extract and persist raw triplets for an already persisted source."""
    database = DatabaseClient(settings.database)
    try:
        async with LocalModelRuntime(settings.local_models) as local_models:
            return await _extract_triplets_with_resources(
                settings,
                source_uuid,
                local_models,
                database,
            )
    finally:
        await database.close()


async def run_semantic_stage(
    settings: Settings,
    source_uuid: str,
) -> SemanticStageResult:
    """Run the complete semantic stage for an existing source."""
    database = DatabaseClient(settings.database)
    try:
        async with LocalModelRuntime(settings.local_models) as local_models:
            return await _run_semantic_stage_with_resources(
                settings,
                source_uuid,
                local_models,
                database,
            )
    finally:
        await database.close()


async def ingest_and_extract_triplets(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceProcessingResult:
    """Ingest a source and extract its raw triplets in one runtime session."""
    database = DatabaseClient(settings.database)
    try:
        async with LocalModelRuntime(settings.local_models) as local_models:
            ingestion = await _ingest_source_with_resources(
                settings,
                pdf_path,
                pages,
                local_models,
                database,
            )
            triplet_count = await _extract_triplets_with_resources(
                settings,
                ingestion.source.uuid,
                local_models,
                database,
            )
    finally:
        await database.close()

    return SourceProcessingResult(
        source=ingestion.source,
        page_count=ingestion.page_count,
        triplet_count=triplet_count,
    )


async def ingest_and_run_semantic_stage(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceSemanticStageResult:
    """Ingest a source and run its complete semantic stage in one session."""
    database = DatabaseClient(settings.database)
    try:
        async with LocalModelRuntime(settings.local_models) as local_models:
            ingestion = await _ingest_source_with_resources(
                settings,
                pdf_path,
                pages,
                local_models,
                database,
            )
            semantic = await _run_semantic_stage_with_resources(
                settings,
                ingestion.source.uuid,
                local_models,
                database,
            )
    finally:
        await database.close()

    return SourceSemanticStageResult(
        source=ingestion.source,
        page_count=ingestion.page_count,
        semantic=semantic,
    )


__all__ = [
    'SemanticStageResult',
    'SourceIngestionResult',
    'SourceProcessingResult',
    'SourceSemanticStageResult',
    'extract_triplets',
    'ingest_and_extract_triplets',
    'ingest_and_run_semantic_stage',
    'ingest_source',
    'list_sources',
    'run_semantic_stage',
]
