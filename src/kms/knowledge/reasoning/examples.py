"""Demonstrations for complete, premise-supported knowledge reasoning."""

from datetime import UTC, datetime

import dspy

from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewArgument,
    NewProposition,
    NewPropositionReference,
)
from kms.knowledge.models import KnowledgeContext, Proposition
from kms.knowledge.reasoning.models import ReasoningCandidates, ReasoningRequest


def reasoning_examples() -> list[dspy.Example]:
    """Show explicit multi-premise support and abstention without a bridge."""
    canadian_student = Proposition(
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        uuid='p1',
        name='Canadian student',
        statement='Alex is a Canadian student.',
    )
    domestic_tuition_rule = Proposition(
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        uuid='p2',
        name='Tuition rule',
        statement='Every Canadian student pays domestic tuition.',
    )
    canada_offers_tuition = Proposition(
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        uuid='p4',
        name='Tuition availability',
        statement='Canadian universities offer domestic tuition.',
    )
    learner_is_canadian = Proposition(
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        uuid='p5',
        name='Canadian learner',
        statement='Sam is a Canadian learner.',
    )

    return [
        dspy.Example(
            request=ReasoningRequest(
                understanding='Does Alex pay domestic tuition?',
                context=KnowledgeContext(
                    matched=[canadian_student],
                    propositions=[canadian_student, domestic_tuition_rule],
                    arguments=[],
                ),
            ),
            candidates=ReasoningCandidates(
                new_propositions=[
                    NewProposition(
                        id='domestic_tuition',
                        name='Alex’s tuition status',
                        statement='Alex pays domestic tuition.',
                    )
                ],
                new_arguments=[
                    NewArgument(
                        premises=[
                            ExistingPropositionReference(uuid='p1'),
                            ExistingPropositionReference(uuid='p2'),
                        ],
                        conclusion=NewPropositionReference(
                            kind='new', id='domestic_tuition'
                        ),
                    )
                ],
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=ReasoningRequest(
                understanding='Does Sam pay domestic tuition?',
                context=KnowledgeContext(
                    matched=[learner_is_canadian],
                    propositions=[canada_offers_tuition, learner_is_canadian],
                    arguments=[],
                ),
            ),
            candidates=ReasoningCandidates(
                new_propositions=[], new_arguments=[]
            ),
        ).with_inputs('request'),
    ]
