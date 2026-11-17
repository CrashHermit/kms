"""Demonstrations for source-grounded proposition and argument extraction."""

import dspy

from kms.knowledge.candidates import (
    NewArgument,
    NewProposition,
    NewPropositionReference,
)
from kms.knowledge.extraction.models import (
    UnitExtractionCandidates,
    UnitExtractionRequest,
)
from kms.media.models import Material
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import SourceUnitMaterial


def extraction_examples() -> list[dspy.Example]:
    """Show complete source-grounded arguments and related claims without a link."""
    examples = [
        (
            'How can a visitor enter the archive?',
            'The archive admits people with a valid pass. Mira has a valid pass. '
            'Therefore Mira can enter the archive.',
            [
                NewProposition(
                    id='rule',
                    name='Admission rule',
                    statement='The archive admits people with a valid pass.',
                ),
                NewProposition(
                    id='pass',
                    name='Mira’s pass',
                    statement='Mira has a valid pass.',
                ),
                NewProposition(
                    id='entry',
                    name='Mira’s admission',
                    statement='Mira can enter the archive.',
                ),
            ],
            [
                NewArgument(
                    premises=[
                        NewPropositionReference(id='rule'),
                        NewPropositionReference(id='pass'),
                    ],
                    conclusion=NewPropositionReference(id='entry'),
                )
            ],
        ),
        (
            'How do daylight and seasons relate?',
            'Daylight duration changes during the year. Temperatures also vary by season.',
            [
                NewProposition(
                    id='daylight',
                    name='Daylight duration',
                    statement='Daylight duration changes during the year.',
                ),
                NewProposition(
                    id='temperature',
                    name='Seasonal temperature',
                    statement='Temperatures vary by season.',
                ),
            ],
            [],
        ),
    ]
    return [
        dspy.Example(
            request=UnitExtractionRequest(
                understanding=understanding,
                context_before=[],
                target=SourceUnitMaterial(
                    uuid='example-unit',
                    kind=SourceUnitKind.CONTENT,
                    material=Material.from_text(text),
                ),
                context_after=[],
                existing_propositions=[],
                existing_arguments=[],
            ),
            candidates=UnitExtractionCandidates(
                new_propositions=propositions,
                existing_proposition_uuids=[],
                new_arguments=arguments,
                existing_argument_uuids=[],
            ),
        ).with_inputs('request')
        for understanding, text, propositions, arguments in examples
    ]
