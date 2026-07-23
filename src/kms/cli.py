"""Command-line entry point for the KMS pipeline: ``python -m kms.cli book.pdf out/``."""

import asyncio
import logging
import sys

from kms.pipeline import run

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pdf = args[0] if args else "test.pdf"
    out_dir = args[1] if len(args) > 1 else "output"
    written = asyncio.run(run(pdf, output_dir=out_dir))
    logger.info("Wrote assembled document to: %s", written)


if __name__ == "__main__":
    main()
