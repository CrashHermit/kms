"""DSPy module for source-content representation formatting."""

import dspy

from kms2.core.model.source_processing.formatting import (
    FormattingInput,
    FormattingOutput,
)


class FormatterSignature(dspy.Signature):
    r"""Normalize each complete source block in a representation-only rewrite.

    Process indexed blocks independently. Return one complete formatted_content
    for every input position, preserving its exact index. Never merge, split,
    omit, or reorder blocks, or borrow content from another block.

    Preserve all source wording, numbers, identifiers, mathematical values,
    line order, paragraph boundaries, and intentional answer blanks. Do not
    solve, simplify, infer answers, paraphrase, or complete truncated content.
    Change only mathematical representation and clearly identified Markdown
    structure. Preserve list numbering and subpart ownership.

    Represent complete mathematical spans in LaTeX, including their variables,
    subscripts, superscripts, operators, relations, functions, and equations.
    Use dollar delimiters for inline mathematics and double-dollar delimiters
    for display mathematics, converting LaTeX bracket delimiters only when
    they delimit actual mathematics. Convert mathematical Unicode notation
    to LaTeX without changing the expression's structure or value. Preserve
    the source's upright versus italic symbol convention.

    Do not introduce fraction or number-set notation when the source uses
    another representation. Leave page/exercise numbers, years, references,
    proper names, standalone nonmathematical values, and uncertain spans
    unchanged. Escape a currency dollar sign as a literal dollar, rather than
    treating a price as mathematics.

    Preserve fenced code, inline code, and image/link destinations verbatim,
    including any literal delimiter characters. Normalize only clear Markdown
    headings, list-marker decoration, emphasis, and tables. An underlined
    heading becomes one hash-marked heading line without losing its title.
    Keep prose punctuation outside math delimiters; punctuation that is
    actually part of a mathematical expression remains inside it.
    """

    blocks: list[FormattingInput] = dspy.InputField(
        description='Ordered complete source blocks with zero-based batch-local positions.'
    )
    formatted_blocks: list[FormattingOutput] = dspy.OutputField(
        description='One representation-formatted complete block for every input position.'
    )


class FormatterModule(dspy.Module):
    """Run one non-recording full-block formatting prediction."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    @staticmethod
    def _validated_outputs(
        blocks: list[FormattingInput],
        formatted_blocks: list[FormattingOutput],
    ) -> list[FormattingOutput]:
        outputs = [
            FormattingOutput.model_validate(block) for block in formatted_blocks
        ]
        input_positions = [block.position for block in blocks]
        expected_positions = set(input_positions)
        if len(input_positions) != len(expected_positions):
            raise ValueError('formatter input positions must be unique')
        positions = [block.position for block in outputs]
        if (
            len(positions) != len(expected_positions)
            or set(positions) != expected_positions
        ):
            raise ValueError(
                'formatter output positions must cover each input position exactly once'
            )
        outputs_by_position = {block.position: block for block in outputs}
        return [
            outputs_by_position[position]
            for position in (block.position for block in blocks)
        ]

    def forward(
        self, *, blocks: list[FormattingInput]
    ) -> list[FormattingOutput]:
        """Format a batch of independently indexed blocks synchronously."""
        prediction = self.predictor(blocks=blocks)
        return self._validated_outputs(blocks, prediction.formatted_blocks)

    async def aforward(
        self, *, blocks: list[FormattingInput]
    ) -> list[FormattingOutput]:
        """Format a batch of independently indexed blocks asynchronously."""
        prediction = await self.predictor.acall(blocks=blocks)
        return self._validated_outputs(blocks, prediction.formatted_blocks)
