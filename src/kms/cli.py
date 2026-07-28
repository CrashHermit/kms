"""Command-line entry point for the KMS pipeline: ``python -m kms.cli book.pdf out/``.

Logging: every stage logs one INFO line summarising what it produced, and one DEBUG line
per DSPy call with that call's inputs' shape and its (elided) outputs. INFO is the default;
set ``KMS_LOG_LEVEL=DEBUG`` for the per-call detail. The stage loggers are named after their
modules (``kms.entity.pedagogical_component_finder``, …), so a single stage can be turned up on its own.
"""

import asyncio
import logging
import os
import sys

from kms import pipeline

logger = logging.getLogger(__name__)

LOG_LEVEL_ENV = 'KMS_LOG_LEVEL'


def _log_level() -> int:
    """The configured log level, defaulting to INFO for an unset or unrecognised value."""
    name = (os.environ.get(LOG_LEVEL_ENV) or 'INFO').strip().upper()
    level = logging.getLevelNamesMapping().get(name)
    return level if isinstance(level, int) else logging.INFO


def main(argv: list[str] | None = None) -> None:
    """Run the pipeline from the command line: ``python -m kms.cli <pdf> [out_dir]``.

    Args:
        argv: Argument list. Reads ``sys.argv[1:]`` when None.
    """
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=_log_level(), format='%(name)s: %(message)s')
    pdf = args[0] if args else 'test.pdf'
    out_dir = args[1] if len(args) > 1 else 'output'
    written = asyncio.run(pipeline.run(pdf, output_dir=out_dir))
    logger.info('Wrote assembled document to: %s', written)


if __name__ == '__main__':
    main()
