"""Source assertions versus requested constructions in proposition extraction."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms2.core.model.source_extraction import (
    SourceExtractionInput,
)
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourcePropositionCandidate,
)


def _targets(*contents: str) -> list[PositionedSourceBlock]:
    return [
        PositionedSourceBlock(
            position=position,
            source_block=SourceBlockContext(
                block_type=BlockType.PARAGRAPH, content=content
            ),
        )
        for position, content in enumerate(contents)
    ]


def source_proposition_extraction_examples() -> list[dspy.Example]:
    """Keep task constraints out while admitting explicit hints and assertions."""
    return [
        dspy.Example(
            request=SourceExtractionInput(
                context_before=[
                    SourceBlockContext(
                        block_type=BlockType.PARAGRAPH,
                        content='Exercise 8. Sketch a graph with the following properties. The conditions continue on the next page.',
                    )
                ],
                target_blocks=_targets(
                    r'$h(2)=7$ and $\lim_{x\to2}h(x)=4$.',
                    r'$h(5)$ is undefined and $\lim_{x\to5}h(x)$ does not exist.',
                    r'a. Estimate the limit of $q(x)=\frac{|x-5|}{x-5}$ as $x$ approaches $a=5$ from nearby values. '
                    r'b. Use algebra to evaluate the limit if it exists, or explain why it does not. '
                    r'(Hint: $|a|=a$ whenever $a\ge0$, but $|a|=-a$ whenever $a<0$.) '
                    r'c. True or false: $q(5)=1$? Explain.',
                ),
                context_after=[],
            ),
            propositions=[
                SourcePropositionCandidate(
                    text=r'For a nonnegative argument $a$ of the absolute-value function, $|a|=a$.',
                    source_position=2,
                ),
                SourcePropositionCandidate(
                    text=r'For a negative argument $a$ of the absolute-value function, $|a|=-a$.',
                    source_position=2,
                ),
            ],
        ).with_inputs('request'),
        dspy.Example(
            request=SourceExtractionInput(
                context_before=[],
                target_blocks=_targets(
                    r'There exists a differentiable real-valued function $f$ on $\mathbb{R}$ with $f(0)=0$ and $f^{\prime}(0)=1$.',
                    r'Find a differentiable function $k$ with $k(0)=3$ and $k^{\prime}(0)=2$.',
                    r'For every real number $x$, $|x|\ge0$.',
                    'The absolute value of any real number is nonnegative.',
                ),
                context_after=[],
            ),
            propositions=[
                SourcePropositionCandidate(
                    text=r'There exists a differentiable real-valued function $f$ on $\mathbb{R}$ with $f(0)=0$ and $f^{\prime}(0)=1$.',
                    source_position=0,
                ),
                SourcePropositionCandidate(
                    text=r'For every real number $x$, $|x|\ge0$.',
                    source_position=2,
                ),
                SourcePropositionCandidate(
                    text='The absolute value of any real number is nonnegative.',
                    source_position=3,
                ),
            ],
        ).with_inputs('request'),
    ]
