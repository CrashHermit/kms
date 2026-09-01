"""Small, focused line-replacement formatting stages."""

import dspy

from kms.core import edits, models, module


class MathFormattingSignature(dspy.Signature):
    r"""Normalize mathematical representation without changing source content.

    Convert `\(...\)` to `$...$`, `\[...\]` to `$$...$$`, and Unicode
    mathematical notation to LaTeX. Preserve every word, number, symbol,
    equation, line, and paragraph boundary. Never solve, simplify, reorder,
    add, or remove content.

    Input records are ordered source lines. Their `index` values are one-based
    and are the only valid coordinates. Copy indexes only from the supplied
    records; never derive an index from numbers in the text. Numbers in `text`
    are content, not coordinates. Return only changed lines. Each replacement
    must contain the complete replacement text for its original line; an empty
    replacement deletes that line. Return an empty list when no change is
    needed.

    Positive example:
    Input: [{"index": 1, "text": "The value is \\(x^2\\)."}]
    Output: [{"index": 1, "replacement": "The value is $x^2$."}]

    Negative example:
    Input: [{"index": 1, "text": "Output Format"}]
    Output: []

    Never output an index that is not present in the input. Copy the index
    from the source record verbatim. If there is one input line, the only
    valid replacement index is 1; numbers such as years, citations, and
    equation values are text content and must not be used as indexes.

    For a one-line input, index 2 is invalid and must never be returned.
    """

    lines: list[models.LineInput] = dspy.InputField(
        description='Ordered source line records with explicit one-based indexes.'
    )
    replacements: list[edits.LineReplacement] = dspy.OutputField(
        description='Replacement-only edits for changed source lines.'
    )


class StructureFormattingSignature(dspy.Signature):
    r"""Normalize Markdown structure without changing source content.

    Normalize only clearly identified headings, list markers, part-marker
    decoration, emphasis, and table structure. Preserve every word, number,
    symbol, mathematical expression, identifier, line order, and paragraph
    boundary. Never add, remove, split, merge, or invent content.

    Input records are ordered source lines. Their `index` values are one-based
    and are the only valid coordinates. Numbers in `text` are content, not
    coordinates. Return only changed lines. Each replacement must contain the
    complete replacement text for its original line; an empty replacement
    deletes that line. Return an empty list when no change is needed.

    Positive example:
    Input: [{"index": 1, "text": "# Heading"}]
    Output: []

    Negative example:
    Input: [{"index": 1, "text": "2. x + 1"}]
    Output: []

    For a one-line input, index 2 is invalid and must never be returned.
    """

    lines: list[models.LineInput] = dspy.InputField(
        description='Ordered source line records with explicit one-based indexes.'
    )
    replacements: list[edits.LineReplacement] = dspy.OutputField(
        description='Replacement-only edits for changed source lines.'
    )


class _BlockRewriter(module.Module):
    def __init__(self, signature, record_name: str, language_model, recorder=None):
        self.signature = signature
        self.record_name = record_name
        super().__init__(language_model, recorder)

    def encode(self, node_content: str) -> dict:
        return {'lines': edits.numbered_lines(node_content)}

    def decode(self, prediction, **inputs) -> str:
        source = inputs['node_content']
        try:
            replacements = [
                edits.LineReplacement.model_validate(replacement)
                for replacement in module.as_list(prediction.replacements)
            ]
            return edits.apply_line_replacements(source, replacements)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ValueError(
                f'{self.record_name} returned invalid replacements: {exc}; '
                f'lines={edits.numbered_lines(source)!r}'
            ) from exc


class MathFormatter:
    def __init__(self, language_model, recorder=None):
        self.editor = _BlockRewriter(
            MathFormattingSignature,
            'math_formatter',
            language_model,
            recorder,
        )

    async def aforward(self, *, node_content: str) -> str:
        return await self.editor.aforward(node_content=node_content)


class StructureFormatter:
    def __init__(self, language_model, recorder=None):
        self.editor = _BlockRewriter(
            StructureFormattingSignature,
            'structure_formatter',
            language_model,
            recorder,
        )

    async def aforward(self, *, node_content: str) -> str:
        return await self.editor.aforward(node_content=node_content)
