"""Compare staged and single-step formatting on representative source material."""

import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault(
    'KMS_MODELS__MODULES__FORMATTER__BASE_URL', 'http://127.0.0.1:8080/v1'
)
os.environ.setdefault(
    'KMS_MODELS__MODULES__FORMATTER__MODEL', 'openai/gemma-4-e4b-qat-text'
)
os.environ.setdefault('KMS_MODELS__MODULES__FORMATTER__API_KEY', 'not-needed')
os.environ.setdefault('KMS_SERVING__MANAGE', '1')

from kms.construction import formatter, formatting_stages  # noqa: E402
from kms.core import llm, serve  # noqa: E402

MATERIAL = [
    r"""2.3. THE QUADRATIC FORMULA 117
Given \(ax^2 + bx + c = 0\), the roots are
x = (-b ± √(b² - 4ac)) / 2a.""",
    r"""The feasible region is defined by
\begin{aligned}
2x + y &\le 10,\\
x - y &\ge 1.
\end{aligned}
""",
    r"""Exercises
---------
ⓐ Evaluate 9x + 7 when x = 3.
ⓑ Compute \(|-25|\) and explain why the result is positive.
ⓒ A book costs $15, and a second book costs $20.""",
    r"""| variable | meaning | value |
|---|---|---|
| R₁ | radius | 2 ≤ r ≤ 5 |
| A | area | πr² |""",
    r"""**Example 4.** Let R₁ × R₂ contain the points
![12]()
Use `x² = 9` as the input.
```python
result = x²
assert result == 9
```""",
    r"""Theorem 2¹ was proved in 2024, on page 319.
Its conclusion is \(f(x) = x^2\), not a new theorem number.""",
    r"""A function is increasing when x₁ < x₂ implies f(x₁) ≤ f(x₂).
For the function f(x) = x², compare f(-2) and f(3).""",
    r"""The degree formula is
\[
\sum_{v \in V} d(v) = 2e.
\]
Thus a graph with degrees 4, 4, 3, 3, 3, 2, 1 has degree sum 20.""",
    r"""Definition 5.1. Let \(A \subseteq B\). Then
\[
\forall x \in A,\; x \in B.
\]
The symbol ⓐ labels the first exercise part; do not interpret it as math.""",
    r"""A proof sketch:
1) Start with \(x_0 = 1\).
2) Apply xₙ₊₁ = (xₙ + 2/xₙ) / 2.
3) Stop when |xₙ² - 2| < 10⁻⁶.
Do not alter the identifier Newton's method.""",
]
OUTPUT = Path('output/live_formatter_comparison.json')


async def main() -> None:
    """Run both formatter designs on the same source blocks."""
    language_model = llm.module_lm('formatter')
    staged = formatter.Formatter(language_model=language_model)
    full_rewrite = formatting_stages.FullRewriteFormatter(language_model)
    comparisons = []
    manager = serve.RouterManager(serve.default_router())
    with serve.model_manager_context(manager):
        for index, content in enumerate(MATERIAL):
            started = perf_counter()
            staged_output = await staged.aforward(node_content=content)
            staged_duration_ms = round((perf_counter() - started) * 1000, 2)
            started = perf_counter()
            rewrite_output = await full_rewrite.aforward(content=content)
            rewrite_duration_ms = round((perf_counter() - started) * 1000, 2)
            comparisons.append(
                {
                    'index': index,
                    'input': content,
                    'staged_output': staged_output,
                    'full_rewrite_output': rewrite_output,
                    'staged_duration_ms': staged_duration_ms,
                    'full_rewrite_duration_ms': rewrite_duration_ms,
                    'outputs_equal': staged_output == rewrite_output,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'saved: {OUTPUT}')
    for comparison in comparisons:
        print(
            f'=== material {comparison["index"]} '
            f'equal={comparison["outputs_equal"]} '
            f'staged_ms={comparison["staged_duration_ms"]} '
            f'full_rewrite_ms={comparison["full_rewrite_duration_ms"]} ==='
        )
        print('--- input ---')
        print(comparison['input'])
        print('--- staged ---')
        print(comparison['staged_output'])
        print('--- full rewrite ---')
        print(comparison['full_rewrite_output'])


if __name__ == '__main__':
    asyncio.run(main())
