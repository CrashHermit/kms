"""Interactive terminal entry point for KMS2 source ingestion."""

import asyncio
import logging
import sys
from pathlib import Path

from InquirerPy import inquirer

from kms2.application import ingest_source
from kms2.config import Settings

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
    """Prompt for a PDF and run the KMS2 source pipeline."""
    pdf_path = inquirer.filepath(
        message='Select the PDF to process:',
        default=str(Path.cwd()),
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
    proceed = inquirer.confirm(
        message='Ingest the document with these settings?',
        default=True,
    ).execute()

    if not proceed:
        logger.info('Cancelled.')
        return

    result = asyncio.run(ingest_source(Settings(), pdf_path, pages))
    logger.info(
        'Done: source %s (%s), %d page(s) persisted.',
        result.source.uuid,
        result.source.key,
        result.page_count,
    )


if __name__ == '__main__':
    run()
