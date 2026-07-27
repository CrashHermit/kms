"""
Logging helpers shared by the pipeline's stages.

Every stage logs through the standard library under its own module name
(``logging.getLogger(__name__)`` → ``kms.entity.pedagogical_component_finder``, …), so a run can be
filtered per stage without any custom machinery. This module holds only the formatting
helpers that would otherwise be copy-pasted into each one.

Two levels, used consistently across the stages:

* ``INFO`` — one line per stage per run: what it was given and what it produced. This is
  the level ``cli`` enables by default, and it is meant to make a run's shape readable
  (how many nodes, how many blocks, how many procedures) without flooding the terminal.
* ``DEBUG`` — one line per *DSPy call*: the inputs' shape and the outputs, elided. This is
  what makes an individual stage's decisions inspectable when something looks wrong.

DELIBERATELY NOT TRACING. The retired ``core.tracing`` wrote structured per-call JSONL that
the ``training/*`` loaders consumed as DSPy examples. This is a debugging aid, not that:
it is lossy (content is elided), unstructured, and goes to a log stream rather than a file.
Restoring trainable capture is a separate decision — see ``docs/HANDOFF.md``.
"""

# Default budget for an elided snippet. Long enough to identify a block by its opening
# words, short enough that a DEBUG line stays one terminal row.
SNIPPET_LIMIT = 80


def elide(value: object, limit: int = SNIPPET_LIMIT) -> str:
    """Return `value` as a single-line string, truncated to `limit` characters.

    Log lines carry markdown and LaTeX pulled straight from the page, which is often long
    and frequently spans lines; both would wreck a log stream. Whitespace (including
    newlines) is collapsed to single spaces and the result is cut at `limit`, with a
    trailing '…' when anything was dropped.

    Args:
        value: The content to render. None and non-strings are coerced.
        limit: Maximum characters to keep.

    Returns:
        The collapsed, truncated single-line form.
    """
    text = ' '.join(str(value or '').split())
    return text if len(text) <= limit else f'{text[:limit]}…'


def counts(items: list[str | None]) -> str:
    """Return a compact ``'a=2 b=1'`` histogram of the given values, most common first.

    Used for the stages that produce an open or enumerated vocabulary (node types, induced
    block types), where the distribution is the useful summary rather than the values.
    Empty input renders as ``'none'``.

    Args:
        items: The values to tally. None renders as '?'.

    Returns:
        The histogram as a single space-separated string.
    """
    tally: dict[str, int] = {}
    for item in items:
        key = item or '?'
        tally[key] = tally.get(key, 0) + 1
    ordered = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    return ' '.join(f'{key}={n}' for key, n in ordered) or 'none'
