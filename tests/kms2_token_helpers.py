"""Deterministic token counters used by KMS2 hub unit tests."""

from kms2.core.windowing import TokenBudget


class FixedCounter:
    """Return one configured token cost for every text in an ordered batch."""

    def __init__(self, cost: int = 1) -> None:
        self.cost = cost

    def count_texts(self, texts: list[str]) -> list[int]:
        return [self.cost] * len(texts)


RERANKER_COUNTER = FixedCounter()
JUDGE_BUDGET = TokenBudget(FixedCounter(), 100000)


class UnexpectedSummary:
    """Reject unexpected summarization in direct-fit behavior fixtures."""

    async def acall(self, *, request):
        raise AssertionError('Direct-fit evidence must not be summarized')


def direct_synthesis(*, include_reranker: bool = True) -> dict[str, object]:
    """Provide isolated synthesis dependencies for direct-fit node fixtures."""
    dependencies: dict[str, object] = {
        'summary_module': UnexpectedSummary(),
        'merge_module': UnexpectedSummary(),
        'final_budget': TokenBudget(FixedCounter(), 100000),
        'summary_budget': TokenBudget(FixedCounter(), 100000),
        'merge_budget': TokenBudget(FixedCounter(), 100000),
    }
    if include_reranker:
        dependencies['reranker_overhead_tokens'] = 0
    return dependencies
