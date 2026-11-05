"""Runtime examples for independently grounded entity-description packets."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionInput,
    SourceEntityDescriptionItem,
    SourceEntityDescriptionOutput,
)


def _block(content: str) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH,
        material=MultimodalContent(text=content),
    )


def source_entity_description_examples() -> list[dspy.Example]:
    """Keep the same symbol's independent meanings scoped to their occurrences."""
    return [
        dspy.Example(
            items=[
                SourceEntityDescriptionItem(
                    position=0,
                    request=SourceEntityDescriptionInput(
                        term='$v$',
                        context_before=[_block('Motion along a straight line')],
                        target_blocks=[
                            _block(
                                'In this passage, $v$ denotes the velocity of the cart.'
                            )
                        ],
                        context_after=[],
                    ),
                ),
                SourceEntityDescriptionItem(
                    position=1,
                    request=SourceEntityDescriptionInput(
                        term='$v$',
                        context_before=[_block('Measuring liquids')],
                        target_blocks=[
                            _block(
                                'The volume of the sample is denoted by $v$.'
                            )
                        ],
                        context_after=[],
                    ),
                ),
                SourceEntityDescriptionItem(
                    position=2,
                    request=SourceEntityDescriptionInput(
                        term='$q$',
                        context_before=[],
                        target_blocks=[
                            _block('Consider the expression $q^2 + 1$.')
                        ],
                        context_after=[],
                    ),
                ),
            ],
            content_images=[],
            descriptions=[
                SourceEntityDescriptionOutput(
                    position=0,
                    description='The symbol $v$ denotes the velocity of the cart in this passage.',
                ),
                SourceEntityDescriptionOutput(
                    position=1,
                    description='The symbol $v$ denotes the volume of the sample in this passage.',
                ),
                SourceEntityDescriptionOutput(
                    position=2,
                    description='The symbol $q$ appears squared in the expression $q^2 + 1$; its referent is not specified.',
                ),
            ],
        ).with_inputs('items', 'content_images')
    ]
