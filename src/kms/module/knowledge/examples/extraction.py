"""Grounded demonstrations for knowledge extraction."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.knowledge.admission import (
    ArgumentDraft,
    DraftPropositionReference,
    ExistingPropositionReference,
    KnowledgeDraft,
    KnowledgeEvidenceBlock,
    KnowledgeExistingProposition,
    KnowledgeExtractionInput,
    KnowledgeResponseSegment,
    PropositionDraft,
)


def _block(position: int, content: str) -> KnowledgeEvidenceBlock:
    return KnowledgeEvidenceBlock(
        position=position,
        source_key='Linear maps',
        page_indexes=[0],
        source_block=SourceBlockContext(
            block_type=BlockType.PARAGRAPH,
            material=MultimodalContent(text=content),
        ),
    )


def knowledge_extraction_examples() -> list[dspy.Example]:
    """Show grounded derivation, later premise reuse, and empty admission."""
    first_request = KnowledgeExtractionInput(
        segments=[
            KnowledgeResponseSegment(
                material=MultimodalContent(
                    text=(
                        'Linearity means T is additive and homogeneous. Using '
                        'T(ax)=aT(x) and T(x+y)=T(x)+T(y), we obtain '
                        'T(ax+y)=aT(x)+T(y). Applying additivity again gives '
                        'T(ax+y+z)=aT(x)+T(y)+T(z).'
                    )
                ),
                block_positions=[0, 1],
                image_positions=[],
                proposition_positions=[0],
            )
        ],
        blocks=[
            _block(0, 'For a linear map T, T(x+y)=T(x)+T(y).'),
            _block(1, 'For a linear map T and scalar a, T(ax)=aT(x).'),
        ],
        existing_propositions=[
            KnowledgeExistingProposition(
                position=0,
                material=MultimodalContent(
                    text='For a linear map T and scalar a, T(ax)=aT(x).'
                ),
            )
        ],
    )
    first_result = KnowledgeDraft(
        propositions=[
            PropositionDraft(
                content='For a linear map T, T(x+y)=T(x)+T(y).',
                supporting_block_positions=[0],
            ),
            PropositionDraft(
                content=(
                    'For a linear map T and scalar a, T(ax+y)=aT(x)+T(y).'
                ),
                supporting_block_positions=[],
            ),
            PropositionDraft(
                content=(
                    'For a linear map T and scalar a, '
                    'T(ax+y+z)=aT(x)+T(y)+T(z).'
                ),
                supporting_block_positions=[],
            ),
        ],
        arguments=[
            ArgumentDraft(
                explanation=(
                    'Apply additivity to ax+y, then homogeneity to the ax term.'
                ),
                premises=[
                    ExistingPropositionReference(kind='existing', position=0),
                    DraftPropositionReference(kind='draft', position=0),
                ],
                conclusion=DraftPropositionReference(kind='draft', position=1),
                supporting_block_positions=[0],
            ),
            ArgumentDraft(
                explanation='Apply additivity once more to the final sum.',
                premises=[
                    DraftPropositionReference(kind='draft', position=1),
                    DraftPropositionReference(kind='draft', position=0),
                ],
                conclusion=DraftPropositionReference(kind='draft', position=2),
                supporting_block_positions=[0],
            ),
        ],
    )
    later_request = KnowledgeExtractionInput(
        segments=[
            KnowledgeResponseSegment(
                material=MultimodalContent(
                    text=(
                        'Using the previously established value for T(ax+y) and '
                        'additivity, T(ax+y+z)=aT(x)+T(y)+T(z).'
                    )
                ),
                block_positions=[0],
                image_positions=[],
                proposition_positions=[0],
            )
        ],
        blocks=[_block(0, 'For a linear map T, T(u+z)=T(u)+T(z).')],
        existing_propositions=[
            KnowledgeExistingProposition(
                position=0,
                material=MultimodalContent(
                    text='For a linear map T and scalar a, T(ax+y)=aT(x)+T(y).'
                ),
            )
        ],
    )
    later_result = KnowledgeDraft(
        propositions=[
            PropositionDraft(
                content=(
                    'For a linear map T and scalar a, '
                    'T(ax+y+z)=aT(x)+T(y)+T(z).'
                ),
                supporting_block_positions=[],
            )
        ],
        arguments=[
            ArgumentDraft(
                explanation=(
                    'Apply additivity to the established value with the new term z.'
                ),
                premises=[
                    ExistingPropositionReference(kind='existing', position=0)
                ],
                conclusion=DraftPropositionReference(kind='draft', position=0),
                supporting_block_positions=[0],
            )
        ],
    )
    empty_request = KnowledgeExtractionInput(
        segments=[
            KnowledgeResponseSegment(
                material=MultimodalContent(
                    text='Please tell me which example you prefer.'
                ),
                block_positions=[],
                image_positions=[],
                proposition_positions=[],
            )
        ],
        blocks=[],
        existing_propositions=[],
    )
    empty_result = KnowledgeDraft(propositions=[], arguments=[])
    return [
        dspy.Example(
            request=first_request,
            content_images=[],
            result=first_result,
        ).with_inputs('request', 'content_images'),
        dspy.Example(
            request=later_request,
            content_images=[],
            result=later_result,
        ).with_inputs('request', 'content_images'),
        dspy.Example(
            request=empty_request,
            content_images=[],
            result=empty_result,
        ).with_inputs('request', 'content_images'),
    ]
