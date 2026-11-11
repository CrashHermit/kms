"""Curated demonstrations for source formatting and targeted cleanup."""

import dspy

from kms.source.processing.formatting.models import (
    FormattingInput,
    FormattingOutput,
)


def formatting_examples() -> list[dspy.Example]:
    """Return fresh formatting batches with source-fidelity contrasts."""
    code = '```python\npattern = r"\\(literal\\)"\nx = "x²"\n```'
    blank = r'$$\underline{\hspace{2cm}} \leq y \leq \underline{\hspace{2cm}}$$'
    return [
        dspy.Example(
            blocks=[
                FormattingInput(
                    position=0,
                    content='Area\n====\nFind where 2 ≤ x ≤ 5 and y = x².',
                ),
                FormattingInput(
                    position=1,
                    content=r'Given \(x^2 + 1 \le y\), compute 25 - 7.',
                ),
            ],
            formatted_blocks=[
                FormattingOutput(
                    position=0,
                    formatted_content='# Area\nFind where $2 \\le x \\le 5$ and $y = x^2$.',
                ),
                FormattingOutput(
                    position=1,
                    formatted_content=r'Given $x^2 + 1 \le y$, compute $25 - 7$.',
                ),
            ],
        ).with_inputs('blocks'),
        dspy.Example(
            blocks=[
                FormattingInput(position=0, content=code),
                FormattingInput(
                    position=1,
                    content=r'Keep `\(literal\)` unchanged; evaluate \(x + 1\).',
                ),
                FormattingInput(
                    position=2,
                    content='Theorem 2¹ was proved in 2024. The shirt costs $15.',
                ),
            ],
            formatted_blocks=[
                FormattingOutput(position=0, formatted_content=code),
                FormattingOutput(
                    position=1,
                    formatted_content=r'Keep `\(literal\)` unchanged; evaluate $x + 1$.',
                ),
                FormattingOutput(
                    position=2,
                    formatted_content=r'Theorem 2¹ was proved in 2024. The shirt costs \$15.',
                ),
            ],
        ).with_inputs('blocks'),
        dspy.Example(
            blocks=[
                FormattingInput(
                    position=0,
                    content=r'The solution is \(x = (-b \pm \sqrt{b^2 - 4ac}) / 2a\).',
                ),
                FormattingInput(
                    position=1,
                    content='Use f(x): then compare x ≤ 2.',
                ),
                FormattingInput(position=2, content=blank),
                FormattingInput(
                    position=3,
                    content='Upright symbols: R₁ × R₂.',
                ),
                FormattingInput(
                    position=4,
                    content='![12]()\nA diagram for part (b).',
                ),
            ],
            formatted_blocks=[
                FormattingOutput(
                    position=0,
                    formatted_content=r'The solution is $x = (-b \pm \sqrt{b^2 - 4ac}) / 2a$.',
                ),
                FormattingOutput(
                    position=1,
                    formatted_content=r'Use $f(x)$: then compare $x \le 2$.',
                ),
                FormattingOutput(position=2, formatted_content=blank),
                FormattingOutput(
                    position=3,
                    formatted_content=r'Upright symbols: $\text{R}_1 \times \text{R}_2$.',
                ),
                FormattingOutput(
                    position=4,
                    formatted_content='![12]()\nA diagram for part (b).',
                ),
            ],
        ).with_inputs('blocks'),
        dspy.Example(
            blocks=[
                FormattingInput(
                    position=0,
                    content=(
                        'For f(x) = x^2, the derivative is 2x. '
                        'ADVERTISEMENT - Subscribe now: https://exam ple .com/offer '
                        'See the complete lesson: https://example.com/derivatives'
                    ),
                    cleanup_instructions=[
                        "Remove the entire footer beginning 'ADVERTISEMENT - Subscribe now:'; it is a promotional artifact, including its corrupted URL fragment."
                    ],
                )
            ],
            formatted_blocks=[
                FormattingOutput(
                    position=0,
                    formatted_content=(
                        'For $f(x) = x^2$, the derivative is $2x$. '
                        'See the complete lesson: https://example.com/derivatives'
                    ),
                )
            ],
        ).with_inputs('blocks'),
    ]
