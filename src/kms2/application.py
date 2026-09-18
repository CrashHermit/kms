"""Application entry points for running KMS2 pipelines."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from kms2.composition import build_semantic_graph, build_source_graph
from kms2.config.settings import Settings
from kms2.core.model import Source
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_catalog_repository import (
    SourceCatalogRepository,
)
from kms2.local_models import LocalModelRuntime
from kms2.train.recorder import Recorder


@dataclass(frozen=True, slots=True)
class SourceStageResult:
    """Counts materialized by one complete source stage."""

    source: Source
    ocr_page_count: int
    corrected_page_count: int
    formatted_page_count: int
    text_seam_page_count: int
    image_seam_page_count: int
    image_description_page_count: int
    split_page_count: int
    instruction_count: int
    exercise_component_count: int
    pedagogical_component_count: int
    statement_count: int
    procedure_count: int
    embedded_page_count: int


@dataclass(frozen=True, slots=True)
class SemanticStageResult:
    """Counts persisted by one complete semantic stage."""

    source_fact_count: int
    triplet_count: int
    source_entity_description_count: int
    source_event_description_count: int
    source_predicate_description_count: int
    source_statement_description_count: int
    source_procedure_description_count: int
    source_entity_hub_count: int
    source_event_hub_count: int
    source_predicate_hub_count: int
    source_triplet_hub_count: int
    source_statement_hub_count: int
    source_procedure_hub_count: int


@dataclass(frozen=True, slots=True)
class SourcePipelineResult:
    """Summary of source ingestion and its complete semantic stage."""

    source_stage: SourceStageResult
    semantic_stage: SemanticStageResult


@dataclass(frozen=True, slots=True)
class _ApplicationResources:
    """Resources shared by the stages of one application invocation."""

    local_models: LocalModelRuntime
    database: DatabaseClient
    recorder: Recorder | None


@asynccontextmanager
async def _application_resources(
    settings: Settings,
) -> AsyncIterator[_ApplicationResources]:
    """Own database, local-model, and optional recorder resources."""
    database = DatabaseClient(settings.database)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    try:
        await schema.ensure_schema(
            database.session,
            embedding_dimension=settings.local_models.embedding.model.dimension,
        )
        async with LocalModelRuntime(settings.local_models) as local_models:
            yield _ApplicationResources(local_models, database, recorder)
    finally:
        await database.close()


def _source_stage_result(final_state: dict[str, object]) -> SourceStageResult:
    """Convert the final source graph state to its public result."""
    return SourceStageResult(
        source=final_state['source'],
        ocr_page_count=len(final_state['ocr_pages']),
        corrected_page_count=len(final_state['corrected_pages']),
        formatted_page_count=len(final_state['formatted_pages']),
        text_seam_page_count=len(final_state['text_seam_pages']),
        image_seam_page_count=len(final_state['image_seam_pages']),
        image_description_page_count=len(final_state['image_described_pages']),
        split_page_count=len(final_state['split_pages']),
        instruction_count=len(final_state['instructions']),
        exercise_component_count=len(final_state['exercise_components']),
        pedagogical_component_count=len(final_state['pedagogical_components']),
        statement_count=len(final_state['statements']),
        procedure_count=len(final_state['procedures']),
        embedded_page_count=len(final_state['embedded_pages']),
    )


def _semantic_stage_result(
    final_state: dict[str, object],
) -> SemanticStageResult:
    """Convert the final semantic graph state to its public result."""
    return SemanticStageResult(
        source_fact_count=len(final_state['source_facts']),
        triplet_count=len(final_state['triplet_occurrences']),
        source_entity_description_count=final_state[
            'source_entity_description_persisted_count'
        ],
        source_event_description_count=final_state[
            'source_event_description_persisted_count'
        ],
        source_predicate_description_count=final_state[
            'source_predicate_description_persisted_count'
        ],
        source_statement_description_count=final_state[
            'source_statement_description_persisted_count'
        ],
        source_procedure_description_count=final_state[
            'source_procedure_description_persisted_count'
        ],
        source_entity_hub_count=final_state['source_entity_hub_count'],
        source_event_hub_count=final_state['source_event_hub_count'],
        source_predicate_hub_count=final_state['source_predicate_hub_count'],
        source_triplet_hub_count=final_state['source_triplet_hub_count'],
        source_statement_hub_count=final_state['source_statement_hub_count'],
        source_procedure_hub_count=final_state['source_procedure_hub_count'],
    )


async def list_sources(settings: Settings) -> list[Source]:
    """Return persisted sources for interactive source selection."""
    database = DatabaseClient(settings.database)
    try:
        return await SourceCatalogRepository(database.session).list_sources()
    finally:
        await database.close()


async def _run_source_stage(
    settings: Settings,
    resources: _ApplicationResources,
    pdf_path: str,
    pages: list[int] | None,
) -> SourceStageResult:
    """Run the source graph and return its materialized stage summary."""
    source = Source(key=Path(pdf_path).name)
    initial_state = {
        'pdf_path': pdf_path,
        'pages': pages,
        'source': source,
    }
    graph = build_source_graph(
        settings,
        resources.local_models,
        resources.database,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke(initial_state)
    return _source_stage_result(final_state)


async def _run_semantic_stage(
    settings: Settings,
    resources: _ApplicationResources,
    source_uuid: str,
) -> SemanticStageResult:
    """Run one complete semantic graph in dependency order."""
    graph = build_semantic_graph(
        settings,
        resources.local_models,
        resources.database,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return _semantic_stage_result(final_state)


async def ingest_source(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceStageResult:
    """Run the source pipeline and return its materialized stage summary."""
    async with _application_resources(settings) as resources:
        return await _run_source_stage(settings, resources, pdf_path, pages)


async def run_semantic_stage(
    settings: Settings,
    source_uuid: str,
) -> SemanticStageResult:
    """Run the complete semantic stage for an existing source."""
    async with _application_resources(settings) as resources:
        return await _run_semantic_stage(settings, resources, source_uuid)


async def ingest_and_run_semantic_stage(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourcePipelineResult:
    """Ingest a source and run its complete semantic stage in one session."""
    async with _application_resources(settings) as resources:
        source_stage = await _run_source_stage(
            settings,
            resources,
            pdf_path,
            pages,
        )
        semantic_stage = await _run_semantic_stage(
            settings,
            resources,
            source_stage.source.uuid,
        )

    return SourcePipelineResult(
        source_stage=source_stage,
        semantic_stage=semantic_stage,
    )


__all__ = [
    'SemanticStageResult',
    'SourcePipelineResult',
    'SourceStageResult',
    'ingest_and_run_semantic_stage',
    'ingest_source',
    'list_sources',
    'run_semantic_stage',
]
