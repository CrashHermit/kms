"""Log-friendly formatting helpers."""

import time

SNIPPET_LIMIT = 80


def elapsed_ms(start: float) -> float:
    """Rounds elapsed milliseconds from a ``time.perf_counter`` start stamp.

    Args:
        start: A ``time.perf_counter()`` timestamp captured before the span.

    Returns:
        Elapsed wall-clock time in milliseconds, rounded to hundredths.
    """
    return round((time.perf_counter() - start) * 1000, 2)


def elide(value: object, limit: int = SNIPPET_LIMIT) -> str:
    """Collapses whitespace in value and truncates it for log lines.

    Args:
        value: The value to render; None renders as the empty string.
        limit: Maximum characters to keep.

    Returns:
        The elided one-line string.
    """
    text = ' '.join(str(value or '').split())
    return text if len(text) <= limit else f'{text[:limit]}…'


def counts(items: list[str | None]) -> str:
    """Tallies items into a compact ``key=count`` log summary.

    None items tally under ``?``. Keys are ordered by descending count
    then name.

    Args:
        items: The values to tally.

    Returns:
        A space-separated ``key=count`` summary, or ``'none'`` when
        the list is empty.
    """
    tally: dict[str, int] = {}
    for item in items:
        key = item or '?'
        tally[key] = tally.get(key, 0) + 1
    ordered = sorted(tally.items(), key=lambda entry: (-entry[1], entry[0]))
    return ' '.join(f'{key}={count}' for key, count in ordered) or 'none'
