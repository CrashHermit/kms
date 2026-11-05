"""Inclusive input budgets for estimated LLM material."""

from dataclasses import dataclass

from kms.core.tokenization.material import MaterialTokenEstimator


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Configured material estimator and inclusive workload limit."""

    counter: MaterialTokenEstimator
    token_limit: int


def fits_token_budget(*, token_count: int, threshold: int) -> bool:
    """Return whether the measured count is within the inclusive threshold."""
    return token_count <= threshold


class InputBudgetExceeded(ValueError):
    """One atomic input cannot fit within the configured request budget."""
