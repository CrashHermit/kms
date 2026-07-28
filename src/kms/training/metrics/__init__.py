"""Metric functions for DSPy prompt optimisation.

Each metric is a callable with the signature
``metric(example, prediction, trace=None) -> bool | float`` — the contract
DSPy's ``BootstrapFewShot`` and ``MIPROv2`` optimisers expect.

Every metric follows the same pattern: a per-stage judge ``dspy.Signature`` fed
through ``dspy.ChainOfThought`` with a dedicated judge LM (``metric_lm`` for
text stages, ``corrector_judge_lm`` for the vision stage). The judge sees:

* the inputs the stage received,
* the expected output (from the trace — what the stage actually produced
  during the captured run), and
* the predicted output (what the optimiser's compiled candidate produced).

**Discrete-error judges**: every judge lists specific typed errors rather than
producing a float score directly. The score is derived formulaically from the
weighted error count — LLMs are unreliable at producing continuous scores.
Each stage has its own judge signature and error taxonomy in a dedicated
sub-module.

During optimisation (``trace is not None``), scores are binarised at a
threshold so the optimiser gets clean pass/fail signals for demo selection.
At eval time (``trace is None``), continuous scores are returned for
inspection.
"""

from collections.abc import Callable

from kms.training.metrics.corrector import corrector_metric
from kms.training.metrics.extractor import extractor_metric
from kms.training.metrics.instruction_distributor import (
    instruction_distributor_metric,
)
from kms.training.metrics.instruction_finder import (
    instruction_finder_metric,
)
from kms.training.metrics.pedagogical_component_finder import (
    pedagogical_component_finder_metric,
)
from kms.training.metrics.role_typer import role_typer_metric
from kms.training.metrics.seam_merger import seam_merger_metric
from kms.training.metrics.splitter import splitter_metric

# Stage name (as in datasets.examples_by_stage) -> metric function.
STAGE_METRICS: dict[str, Callable] = {
    'corrector': corrector_metric,
    'extractor': extractor_metric,
    'instruction_distributor': instruction_distributor_metric,
    'instruction_finder': instruction_finder_metric,
    'pedagogical_component_finder': pedagogical_component_finder_metric,
    'role_typer': role_typer_metric,
    'seam_merger': seam_merger_metric,
    'splitter': splitter_metric,
}
"""Maps a stage's trace name to its metric, for programmatic lookup."""
