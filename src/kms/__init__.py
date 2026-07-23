"""KMS pipeline package: turns a math textbook PDF into a knowledge graph of math
entities (Definitions, Theorems, Problems), following AutoMathKG (arXiv:2505.13406).

Public API: ``run`` — the async entry point that executes the full pipeline on a PDF.
It is exposed lazily (imported on first access) so importing a submodule such as
``kms.core.state`` does not pull in the heavy graph/LLM stack.
"""

from typing import TYPE_CHECKING

__all__ = ["run"]

if TYPE_CHECKING:
    from kms.pipeline import run


def __getattr__(name: str) -> object:
    if name == "run":
        from kms.pipeline import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
