"""Training and prompt optimisation — metrics, optimiser invocations, and the
bridge from captured traces (``core.tracing`` / ``core.datasets``) to DSPy's
``teleprompt`` optimisers.

The ``KMS_OPTIMIZED_DIR`` env var points at a directory of compiled-stage JSON
files. When set, the pipeline loads them automatically at startup so production
runs pick up the optimised prompts with no code changes."""

import os
from pathlib import Path

import dspy

_OPTIMIZED_DIR_ENV = 'KMS_OPTIMIZED_DIR'


def load_if_exists[M: dspy.Module](
    stage: str,
    module_cls: type[M],
    language_model: dspy.LM,
) -> M:
    """Load an optimised module from disk if present; otherwise return a fresh one.

    Args:
        stage: The stage name (e.g. ``'corrector'``).
        module_cls: The ``dspy.Module`` subclass to instantiate as a fallback.
        lm: The language model to set on the loaded (or fresh) module.

    Returns:
        The loaded optimised module, or a fresh one if no saved file was found.
    """
    optimized_dir = (os.environ.get(_OPTIMIZED_DIR_ENV) or '').strip()
    module = module_cls()
    if optimized_dir:
        path = Path(optimized_dir) / f'{stage}.json'
        if path.exists():
            module.load(str(path))
    module.set_lm(language_model)
    return module