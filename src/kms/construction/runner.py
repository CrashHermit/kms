"""Run document ingestion inside a shared KMS runtime."""

from pathlib import Path

from kms import config, runtime
from kms.construction import workflow
from kms.core import recording


async def ingest(
    application: runtime.Runtime,
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    pages: list[int] | None = None,
    source: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> dict:
    """Invoke the ingestion graph for one document."""
    output_dir = Path(output_dir)
    source = source or Path(pdf_path).name
    recorder = None
    if config.get_settings().recording.enabled:
        recorder = recording.Recorder(
            source,
            output_dir=str(output_dir / 'examples'),
            pdf=str(pdf_path),
            pages=pages,
            title=title,
            author=author,
        )

    graph = workflow.build_workflow(
        recorder=recorder,
        neo4j_session_factory=application.session_factory(),
        neo4j_configured=application.neo4j_configured,
        model_manager=application.model_manager,
    )
    return await graph.ainvoke(
        {
            'pdf_path': str(pdf_path),
            'output_dir': str(output_dir),
            'pages': pages,
            'source_key': source,
            'source_metadata': {'title': title, 'author': author},
        },
        {'recursion_limit': config.get_settings().concurrency.recursion_limit},
    )
