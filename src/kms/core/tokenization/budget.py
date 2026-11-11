"""Inclusive input budgets for estimated LLM material."""

from collections.abc import Callable
from dataclasses import dataclass

from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.models import ImageAsset


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Live workload limit and an optional full model-context measurement."""

    counter: MaterialTokenEstimator
    token_limit: int
    format_request: Callable[[dict[str, object]], str] | None = None
    context_token_limit: int | None = None
    prompt_images: tuple[ImageAsset, ...] = ()


def fits_token_budget(*, token_count: int, threshold: int) -> bool:
    """Return whether the measured count is within the inclusive threshold."""
    return token_count <= threshold


class InputBudgetExceeded(ValueError):
    """One atomic input cannot fit within the configured request budget."""
