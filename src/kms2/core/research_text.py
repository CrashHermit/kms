"""Select complete source spans from provider Markdown without rewriting it."""

from __future__ import annotations

from dataclasses import dataclass

from markdown_it import MarkdownIt

from kms2.core.windowing import TextTokenCounter


@dataclass(frozen=True, slots=True)
class MarkdownSpan:
    """One contiguous source slice suitable for independent reading."""

    line_start: int
    line_end: int
    raw_content: str
    heading_path: tuple[str, ...]
    token_count: int


def _display_math_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    delimiter: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if start is None:
            if stripped.startswith('$$'):
                start = index
                if stripped.count('$$') >= 2:
                    ranges.append((start, index + 1))
                    start = None
                else:
                    delimiter = '$$'
            elif stripped.startswith(r'\['):
                start = index
                delimiter = r'\]'
        elif delimiter == '$$' and '$$' in stripped:
            ranges.append((start, index + 1))
            start = None
            delimiter = None
        elif delimiter == r'\]' and r'\]' in stripped:
            ranges.append((start, index + 1))
            start = None
            delimiter = None
    if start is not None:
        ranges.append((start, len(lines)))
    return ranges


def _heading_text(token: object) -> str:
    children = getattr(token, 'children', None) or []
    return ''.join(
        child.content for child in children if child.type == 'text'
    ).strip()


def markdown_spans(
    markdown: str,
    counter: TextTokenCounter,
    *,
    passage_token_budget: int,
) -> list[MarkdownSpan]:
    """Parse and greedily join complete Markdown constructs into raw spans.

    Source offsets are zero-based and end-exclusive. Every source character is
    retained exactly once across returned spans.
    """
    if not markdown:
        return []

    lines = markdown.splitlines(keepends=True)
    if not lines:
        return []
    tokens = MarkdownIt('commonmark').enable('table').parse(markdown)
    ranges: list[tuple[int, int]] = []
    range_headings: list[tuple[int, int, tuple[str, ...]]] = []
    heading_path: list[str] = []
    for index, token in enumerate(tokens):
        if token.type == 'heading_open':
            level = int(token.tag[1:])
            heading_path = heading_path[: level - 1]
            heading_path.append(_heading_text(tokens[index + 1]))
        if token.map is not None and token.map[1] > token.map[0]:
            bounds = (token.map[0], token.map[1])
            ranges.append(bounds)
            range_headings.append((*bounds, tuple(heading_path)))
    ranges.extend(_display_math_ranges(lines))
    if not ranges:
        ranges = [(0, len(lines))]

    ranges.sort()
    merged_ranges: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            prior_start, prior_end = merged_ranges[-1]
            merged_ranges[-1] = (prior_start, max(prior_end, end))
        else:
            merged_ranges.append((start, end))

    # Attach leading, inter-span, and trailing unclassified lines to neighbors.
    complete_ranges: list[tuple[int, int]] = []
    for index, (_, end) in enumerate(merged_ranges):
        start = 0 if index == 0 else merged_ranges[index - 1][1]
        complete_ranges.append((start, end))
    if complete_ranges:
        start, _ = complete_ranges[-1]
        complete_ranges[-1] = (start, len(lines))
    else:
        complete_ranges.append((0, len(lines)))

    heading_for_span: list[tuple[str, ...]] = []
    for start, end in complete_ranges:
        candidates = [
            path
            for item_start, item_end, path in range_headings
            if item_start < end and item_end > start
        ]
        heading_for_span.append(candidates[-1] if candidates else ())

    raw_spans = [''.join(lines[start:end]) for start, end in complete_ranges]
    costs = counter.count_texts(raw_spans)
    spans: list[MarkdownSpan] = []
    for (start, end), raw, cost, headings in zip(
        complete_ranges, raw_spans, costs, heading_for_span, strict=True
    ):
        if (
            spans
            and spans[-1].token_count <= passage_token_budget
            and cost <= passage_token_budget
        ):
            previous = spans[-1]
            combined = previous.raw_content + raw
            combined_cost = counter.count_texts([combined])[0]
            if combined_cost <= passage_token_budget:
                spans[-1] = MarkdownSpan(
                    previous.line_start,
                    end,
                    combined,
                    headings or previous.heading_path,
                    combined_cost,
                )
                continue
        spans.append(MarkdownSpan(start, end, raw, headings, cost))

    if ''.join(span.raw_content for span in spans) != markdown:
        raise RuntimeError(
            'Research Markdown spans failed source reconstruction'
        )
    return spans
