"""Compose atomic target decomposition."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory
from kms.learning.probe.targets.examples import target_decomposition_examples
from kms.learning.probe.targets.inference import (
    TargetDecompositionModule,
    TargetDecompositionSignature,
)


def build_target_decomposition(
    settings: Settings, *, predictors: PredictorFactory
) -> TargetDecompositionModule:
    """Build decomposition with the configured text inference profile."""
    examples = target_decomposition_examples()
    return TargetDecompositionModule(
        predictors.create(
            TargetDecompositionModule,
            settings.target_decomposition.inference,
            TargetDecompositionSignature,
            examples=examples,
        )
    )
