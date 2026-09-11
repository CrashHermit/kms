"""Application entry points for running KMS2 pipelines."""

from dataclasses import dataclass
from pathlib import Path

from kms2.composition import build_source_graph
from kms2.config import Settings
from kms2.core.model.source import Source
from kms2.database.client import DatabaseClient
from kms2.local_models import LocalModelRuntime


@dataclass(frozen=True, slots=True)
class SourceIngestionResult:
    """Summary of one completed source ingestion."""

    source: Source
    page_count: int


async def ingest_source(
    settings: Settings,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceIngestionResult:
    """Run the source pipeline and return its persisted-source summary."""
    source = Source(key=Path(pdf_path).name)
    initial_state = {
        'pdf_path': pdf_path,
        'pages': pages,
        'source': source,
    }
    database = DatabaseClient(settings.database)

    try:
        async with LocalModelRuntime(settings.local_models) as local_models:
            graph = build_source_graph(
                settings, local_models, database
            ).build_graph()
            final_state = await graph.ainvoke(initial_state)
    finally:
        await database.close()
    return SourceIngestionResult(
        source=final_state['source'],
        page_count=len(final_state['split_pages']),
    )


__all__ = ['SourceIngestionResult', 'ingest_source']
