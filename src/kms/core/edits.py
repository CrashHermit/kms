"""Line-numbered text edits returned by the OCR proofreader."""

from pydantic import BaseModel, Field


class LineEdit(BaseModel):
    """A single replacement of one numbered markdown line.

    Lines are 1-based as displayed to the proofreader; an empty
    replacement deletes the line, and a replacement containing line
    breaks inserts several lines in its place.
    """

    index: int = Field(
        description='The 1-based line number of the line to change.'
    )
    replacement: str = Field(
        description='The full new text of that line. An empty string deletes '
        'the line; a value containing line breaks inserts several lines in '
        'its place.'
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

    Edits are validated first (no duplicate or out-of-range line
    numbers), then applied from highest line number to lowest so that
    earlier indexes stay valid while lines are inserted or deleted.

    Args:
        markdown: The transcription text being corrected.
        edits: The edits to apply, in any order.

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
        lines[edit.index - 1 : edit.index] = replacement_lines
    return '\n'.join(lines)
