"""Interactive terminal entry point for KMS2 source processing."""

import asyncio
import logging
import sys
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from kms2.application import (
    ingest_source,
    list_sources,
    run_semantic_stage,
)
from kms2.config import Settings
from kms2.core.model.source import Source

PDF_DIRECTORY = Path('pdfs')
NEW_SOURCE_OPTION = 'Ingest a new PDF'
EXISTING_SOURCE_OPTION = 'Run the semantic stage on an existing source'


def _pdf_directory() -> Path:
    """Return the project-local directory used for source PDFs."""
    directory = Path.cwd() / PDF_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    return directory


logger = logging.getLogger(__name__)


def run() -> None:
    """Run the KMS2 TUI, exiting cleanly on Ctrl-C."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        _run_tui()
    except KeyboardInterrupt:
        logger.info('Cancelled.')
        sys.exit(0)


def _run_tui() -> None:
    """Prompt for a source and run the selected semantic stage workflow."""
    settings = Settings()
    sources = asyncio.run(list_sources(settings))

    if not sources:
        _run_new_source(settings)
        return

    action = inquirer.select(
        message='What would you like to do?',
        choices=[NEW_SOURCE_OPTION, EXISTING_SOURCE_OPTION],
    ).execute()
    if action == EXISTING_SOURCE_OPTION:
        _run_existing_source(settings, sources)
        return
    _run_new_source(settings)


def _log_semantic_completion(source: Source, semantic) -> None:
    """Log all counts persisted by one complete semantic stage."""
    logger.info(
        'Done: source %s (%s), semantic stage persisted %d assertion(s), '
        '%d entity enrichment(s), %d event enrichment(s), and '
        '%d predicate enrichment(s).',
        source.uuid,
        source.key,
        semantic.raw_assertion_count,
        semantic.entity_enrichment_count,
        semantic.event_enrichment_count,
        semantic.predicate_enrichment_count,
    )


def _run_new_source(settings: Settings) -> None:
    pdf_path = inquirer.filepath(
        message='Select the PDF to process:',
        default=str(_pdf_directory()),
        only_files=True,
    ).execute()
    raw_pages = inquirer.text(
        message=(
            'Limit to pages (0-based, comma-separated, or leave empty for all):'
        ),
        default='',
    ).execute()
    pages = (
        None
        if not raw_pages
        else [int(page.strip()) for page in raw_pages.split(',')]
    )
    result = asyncio.run(ingest_source(settings, pdf_path, pages))
    logger.info(
        'Done: source %s (%s), ingested %d page(s).',
        result.source.uuid,
        result.source.key,
        result.page_count,
    )


def _run_existing_source(
    settings: Settings,
    sources: list[Source],
) -> None:
    source = inquirer.select(
        message='Select the source for the semantic stage:',
        choices=[
            Choice(
                value=source,
                name=f'{source.key} ({source.uuid})',
            )
            for source in sources
        ],
    ).execute()
    proceed = inquirer.confirm(
        message='Run the semantic stage on the selected source?',
        default=True,
    ).execute()

    if not proceed:
        logger.info('Cancelled.')
        return

    semantic = asyncio.run(run_semantic_stage(settings, source.uuid))
    _log_semantic_completion(source, semantic)


if __name__ == '__main__':
    run()
