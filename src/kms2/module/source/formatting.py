"""DSPy module for source-content representation formatting."""

import dspy


def _normalize_math_delimiters(content: str) -> str:
    """Convert LaTeX bracket delimiters to the canonical Markdown form."""
    return (
        content.replace(r'\(', '$')
        .replace(r'\)', '$')
        .replace(r'\[', '$$')
        .replace(r'\]', '$$')
    )


class FormatterSignature(dspy.Signature):
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


class FormatterModule(dspy.Module):
    """Run one non-recording full-block formatting prediction."""

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.predictor = dspy.Predict(FormatterSignature)
        self.predictor.set_lm(language_model)

    def forward(self, *, content: str) -> str:
        """Format one source-content block synchronously."""
        prediction = self.predictor(content=content)
        return _normalize_math_delimiters(prediction.formatted_content)

    async def aforward(self, *, content: str) -> str:
        """Format one source-content block asynchronously."""
        prediction = await self.predictor.acall(content=content)
        return _normalize_math_delimiters(prediction.formatted_content)
