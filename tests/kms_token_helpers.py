"""Deterministic token counters used by KMS2 hub unit tests."""

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator


class FixedCounter:
    """Return one configured token cost for every text in an ordered batch."""

    def __init__(self, cost: int = 1) -> None:
        self.cost = cost

    def count_texts(self, texts: list[str]) -> list[int]:
        return [self.cost] * len(texts)


JUDGE_BUDGET = TokenBudget(MaterialTokenEstimator(FixedCounter()), 100000)


class UnexpectedSummary:
    """Reject unexpected summarization in direct-fit behavior fixtures."""

    async def acall(self, *, request):
        raise AssertionError('Direct-fit evidence must not be summarized')


def direct_synthesis() -> dict[str, object]:
    """Provide isolated synthesis dependencies for direct-fit node fixtures."""
    return {
        'summary_module': UnexpectedSummary(),
        'merge_module': UnexpectedSummary(),
        'final_budget': TokenBudget(
            MaterialTokenEstimator(FixedCounter()), 100000
        ),
        'summary_budget': TokenBudget(
            MaterialTokenEstimator(FixedCounter()), 100000
        ),
        'merge_budget': TokenBudget(
            MaterialTokenEstimator(FixedCounter()), 100000
        ),
    }
