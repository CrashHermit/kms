"""Small, focused line-replacement formatting stages."""

import re

import dspy

from kms.core import edits, models, module

_MATH_DELIMITER_PATTERN = re.compile(r'\\(?P<delimiter>[()[\]])')
_MATH_DELIMITER_REPLACEMENTS = {
    '(': '$',
    ')': '$',
    '[': '$$',
    ']': '$$',
}


def normalize_math_delimiters(content: str) -> str:
    """Converts escaped math delimiters to the canonical dollar form."""
    return _MATH_DELIMITER_PATTERN.sub(
        lambda match: _MATH_DELIMITER_REPLACEMENTS[match['delimiter']],
        content,
    )


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
    def __init__(
        self, signature, record_name: str, language_model, recorder=None
    ):
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
            return normalize_math_delimiters(
                edits.apply_line_replacements(source, replacements)
            )
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


class FullFormattingSignature(dspy.Signature):
    r"""Normalize one complete Markdown source block in a single rewrite.

    Change representation only. Preserve every word, number, identifier,
    mathematical value, line order, and paragraph boundary. Never solve,
    simplify, reorder, invent, or remove content. Every mathematical span
    must be represented completely in LaTeX: variables, numeric literals,
    subscripts, superscripts, operators, relations, functions, and equations.
    Use `\text{...}` for plain upright variables or symbols when that is the
    source's notation. Do not change an expression's value or rewrite it into
    an equivalent form.

    Convert `\(...\)` to `$...$`, `\[...\]` to `$$...$$`, and bare display
    equations to `$$...$$`. Convert mathematical Unicode notation to LaTeX,
    including superscripts, subscripts, operators, relations, radicals, and
    Greek symbols. Wrap clearly mathematical bare expressions in `$...$`, but
    leave page numbers, exercise numbers, years, labels, lone values, and
    uncertain spans unchanged. A currency amount is not mathematics: write
    `$15` as `\$15`, never as `$\$15$`.

    Normalize only clearly identified Markdown structure: headings, list
    markers, part-marker decoration, emphasis, and tables. Convert an
    underlined heading into one `#`-marked line. Preserve figure placeholders
    such as `![N]()`, fenced code, inline code, proper names, references, and
    all content at the block boundaries.

    POSITIVE EXAMPLES — APPLY ALL REQUIRED CHANGES:

    Input:
    Area
    ====
    Find where 2 ≤ x ≤ 5 and y = x².

    Output:
    # Area
    Find where $2 \le x \le 5$ and $y = x^2$.

    Input:
    Given \(x^2 + 1 \le y\), compute 25 - 7.

    Output:
    Given $x^2 + 1 \le y$, compute $25 - 7$.

    NEGATIVE EXAMPLES — PROTECT THESE BOUNDARIES:

    Input:
    ```
    x² = 9
    ![12]()
    ```

    Output:
    ```
    x² = 9
    ![12]()
    ```

    Input:
    The solution is \(x = (-b \pm \sqrt{b^2 - 4ac}) / 2a\).

    Output:
    The solution is $x = (-b \pm \sqrt{b^2 - 4ac}) / 2a$.

    Input:
    Theorem 2¹ was proved in 2024. The shirt costs $15.

    Output:
    Do not use `\frac` or `\mathbb` unless that notation is already present in
    the source. Do not rewrite a mathematical expression into an equivalent
    form. For example, every part of `x₁ + 2y₂ = 10` is mathematical, so it
    should become `$\text{x}_1 + 2\text{y}_2 = 10$` when plain upright
    variables are the source convention. Likewise, `R₁ × R₂` should become
    `$\text{R}_1 \times \text{R}_2$`. Do not use line numbers, page numbers,
    years, theorem numbers, or currency amounts as evidence that surrounding
    text is mathematics. Return only the complete rewritten block.

    """

    content: str = dspy.InputField(
        description='One canonical source-content block in Markdown.'
    )
    formatted_content: str = dspy.OutputField(
        description='The complete block after representation-only formatting.'
    )


class FullRewriteFormatter(dspy.Module):
    """Run one non-recording full-block formatting prediction."""

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(FullFormattingSignature)
        self.predictor.set_lm(language_model)

    def forward(self, *, content: str) -> str:
        """Format one source-content block synchronously."""
        prediction = self.predictor(content=content)
        return normalize_math_delimiters(prediction.formatted_content)

    async def aforward(self, *, content: str) -> str:
        """Format one source-content block asynchronously."""
        prediction = await self.predictor.acall(content=content)
        return normalize_math_delimiters(prediction.formatted_content)
