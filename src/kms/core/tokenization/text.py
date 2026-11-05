"""Shared text-token counting contracts."""

from collections.abc import Sequence
from typing import Protocol


class TextTokenCounter(Protocol):
    """Count ordered text payloads with the consuming model's tokenizer."""

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return one token count per text, preserving input order."""
        ...


def count_text_tokens(
    texts: Sequence[str | None], counters: Sequence[TextTokenCounter]
) -> list[int]:
    """Return conservative maximum token counts across consuming models."""
    normalized = [text or '' for text in texts]
    if not any(normalized):
        return [0] * len(normalized)
    counts = counters[0].count_texts(normalized)
    for counter in counters[1:]:
        other_counts = counter.count_texts(normalized)
        counts = [
            max(first, second)
            for first, second in zip(counts, other_counts, strict=True)
        ]
    return counts
