"""Deterministic line-addressed operations for OCR text transformations."""

from typing import Literal

from pydantic import BaseModel, Field


class LineEdit(BaseModel):
    """One deterministic operation anchored to an original numbered line.

    ``replacement`` may contain newlines when an operation inserts or
    replaces multiple lines. Empty replacement text is a deletion only for
    the default ``replace`` operation; use ``delete`` when the intent should
    be explicit.
    """

    index: int = Field(
        description='The 1-based line number of the line to change or anchor.'
    )
    replacement: str = Field(
        description='The replacement or inserted text; line breaks insert '
        'multiple lines.'
    )
    operation: Literal['replace', 'delete', 'insert_before', 'insert_after'] = (
        Field(
            default='replace',
            description=(
                'The operation: replace the line, delete it, or insert the '
                'replacement before or after it.'
            ),
        )
    )


def number_lines(markdown: str) -> str:
    """Prefixes each line of markdown with its 1-based line number.

    Args:
        markdown: The transcription text to number.

    Returns:
        The same text with every line prefixed by ``[n] ``.
    """
    return '\n'.join(
        f'[{index + 1}] {line}'
        for index, line in enumerate(markdown.split('\n'))
    )


def apply_line_edits(markdown: str, edits: list[LineEdit]) -> str:
    """Applies a list of line edits to a markdown transcription.

    Edits are validated first (no duplicate or out-of-range anchor line
    numbers), then applied from highest anchor line number to lowest so that
    earlier indexes stay valid while lines are inserted or deleted. Insertion
    operations retain their anchor line; replacement and deletion operate on
    it.

    Args:
        markdown: The transcription text being corrected.
        edits: The edits to apply, in any order. Each edit targets one
            original line and may replace, delete, or insert around it.

    Returns:
        The corrected transcription.

    Raises:
        RuntimeError: If an edit targets a duplicate or out-of-range
            line number.
    """
    lines = markdown.split('\n')
    seen: set[int] = set()
    for edit in edits:
        if edit.index in seen:
            raise RuntimeError(f'duplicate edit for line {edit.index}')
        seen.add(edit.index)
        if not 1 <= edit.index <= len(lines):
            raise RuntimeError(
                f'edit line {edit.index} out of range ({len(lines)} lines)'
            )
    for edit in sorted(edits, key=lambda item: item.index, reverse=True):
        replacement_lines = (
            edit.replacement.split('\n') if edit.replacement else []
        )
        position = edit.index - 1
        if edit.operation == 'replace':
            lines[position : position + 1] = replacement_lines
        elif edit.operation == 'delete':
            lines[position : position + 1] = []
        elif edit.operation == 'insert_before':
            lines[position:position] = replacement_lines
        elif edit.operation == 'insert_after':
            lines[position + 1 : position + 1] = replacement_lines
        else:  # pragma: no cover - Literal/Pydantic validate this
            raise RuntimeError(f'unknown line edit operation: {edit.operation}')
    return '\n'.join(lines)
