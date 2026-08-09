SNIPPET_LIMIT = 80


def elide(value: object, limit: int = SNIPPET_LIMIT) -> str:
    text = ' '.join(str(value or '').split())
    return text if len(text) <= limit else f'{text[:limit]}…'


def counts(items: list[str | None]) -> str:
    tally: dict[str, int] = {}
    for item in items:
        key = item or '?'
        tally[key] = tally.get(key, 0) + 1
    ordered = sorted(tally.items(), key=lambda entry: (-entry[1], entry[0]))
    return ' '.join(f'{key}={count}' for key, count in ordered) or 'none'

