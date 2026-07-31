"""Interactive TUI for the KMS pipeline.

Replaces the old CLI with a guided multi-step TUI powered by InquirerPy.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from InquirerPy import inquirer

from kms import pipeline

logger = logging.getLogger(__name__)

_LOG_LEVEL_ENV = 'KMS_LOG_LEVEL'


def _configure_logging(level_name: str) -> None:
    """Set the log level for all pipeline stages."""
    os.environ[_LOG_LEVEL_ENV] = level_name
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    logging.basicConfig(level=level, format='%(name)s: %(message)s')


def _validate_pages(raw: str) -> bool | str:
    """Return True if *raw* is a valid comma-separated page list."""
    try:
        parts = [int(p.strip()) for p in raw.split(',')]
        if any(part < 0 for part in parts):
            return 'All page numbers must be >= 0'
        return True
    except ValueError:
        return 'Must be comma-separated integers (e.g. "0,5,10")'


def _collect_advanced_options() -> dict:
    """Prompt for optional advanced pipeline parameters.

    Returns:
        Dict with optional keys: pages, source, title, author.
    """
    pages_raw = inquirer.text(
        message=(
            'Limit to pages (0-based, comma-separated, or leave empty for all):'
        ),
        default='',
        validate=lambda value: (
            _validate_pages(value) if value.strip() else True
        ),
    ).execute()

    pages: list[int] | None = None
    if pages_raw.strip():
        pages = [int(p.strip()) for p in pages_raw.split(',')]

    source_raw = inquirer.text(
        message='Source key (defaults to PDF filename if empty):',
        default='',
    ).execute()
    source = source_raw.strip() or None

    title_raw = inquirer.text(
        message='Book title (optional):',
        default='',
    ).execute()
    title = title_raw.strip() or None

    author_raw = inquirer.text(
        message='Book author (optional):',
        default='',
    ).execute()
    author = author_raw.strip() or None

    return {
        'pages': pages,
        'source': source,
        'title': title,
        'author': author,
    }


def run() -> None:
    """Launch the interactive TUI and execute the pipeline."""
    try:
        _run_tui()
    except KeyboardInterrupt:
        logger.info('Cancelled.')
        sys.exit(0)


def _run_tui() -> None:
    # ── 1. Pick the PDF ───────────────────────────────────────────
    pdf_path = inquirer.filepath(
        message='Select the PDF to process:',
        default=str(Path.cwd()),
        validate=lambda path: (
            True
            if Path(path).suffix.lower() == '.pdf'
            else 'Must be a .pdf file'
        ),
        only_files=True,
    ).execute()

    if not pdf_path:
        logger.info('No PDF selected — exiting.')
        return

    # ── 2. Output directory ───────────────────────────────────────
    out_dir = inquirer.text(
        message='Output directory:',
        default='output',
        validate=lambda path: (
            True if path.strip() else 'Output directory is required'
        ),
    ).execute()

    # ── 3. Advanced options ───────────────────────────────────────
    use_advanced = inquirer.confirm(
        message='Configure advanced options (pages, source, title, author)?',
        default=False,
    ).execute()

    pages: list[int] | None = None
    source: str | None = None
    title: str | None = None
    author: str | None = None

    if use_advanced:
        advanced = _collect_advanced_options()
        pages = advanced['pages']
        source = advanced['source']
        title = advanced['title']
        author = advanced['author']

    # ── 4. Log level ──────────────────────────────────────────────
    log_level = inquirer.select(
        message='Log verbosity:',
        choices=[
            'INFO',
            'DEBUG',
            'WARNING',
        ],
        default='INFO',
    ).execute()

    _configure_logging(log_level)

    # ── 5. Confirm and run ────────────────────────────────────────
    logger.info('PDF: %s', pdf_path)
    logger.info('Output: %s', out_dir)
    if pages:
        logger.info('Pages: %s', pages)
    if source:
        logger.info('Source: %s', source)
    if title:
        logger.info('Title: %s', title)
    if author:
        logger.info('Author: %s', author)
    logger.info('Log level: %s', log_level)

    proceed = inquirer.confirm(
        message='Run the pipeline with these settings?',
        default=True,
    ).execute()

    if not proceed:
        logger.info('Cancelled.')
        return

    markdown = asyncio.run(
        pipeline.run(
            pdf_path,
            output_dir=out_dir,
            pages=pages,
            source=source,
            title=title,
            author=author,
        )
    )
    logger.info('Assembled document (%d characters).', len(markdown))


if __name__ == '__main__':
    run()
