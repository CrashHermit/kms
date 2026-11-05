"""Explicit demonstrations for exact proposition identity judgments."""

import dspy

from kms.core.model.knowledge.admission import (
    PropositionEquivalenceDecision,
    PropositionEquivalenceInput,
    PropositionIdentityCandidate,
)


def proposition_equivalence_examples() -> list[dspy.Example]:
    """Distinguish paraphrase and alpha-renaming from changed propositions."""
    request = PropositionEquivalenceInput(
        proposition_content=('For every real x, x^2 is nonnegative.'),
        candidates=[
            PropositionIdentityCandidate(
                position=0,
                content='Every real number squared is greater than or equal to zero.',
            ),
            PropositionIdentityCandidate(
                position=1,
                content='For every complex x, x^2 is nonnegative.',
            ),
            PropositionIdentityCandidate(
                position=2,
                content='There exists a real x such that x^2 is negative.',
            ),
            PropositionIdentityCandidate(
                position=3,
                content='For every real y, y^2 is nonnegative.',
            ),
            PropositionIdentityCandidate(
                position=4,
                content='For every real x, x^2 is positive.',
            ),
            PropositionIdentityCandidate(
                position=5,
                content=(
                    'In exercise A, f(x)=x^2; in exercise B, f(x)=sin(x). '
                    'These local definitions are not one function identity.'
                ),
            ),
        ],
    )
    decisions = [
        PropositionEquivalenceDecision(position=0, is_equivalent=True),
        PropositionEquivalenceDecision(position=1, is_equivalent=False),
        PropositionEquivalenceDecision(position=2, is_equivalent=False),
        PropositionEquivalenceDecision(position=3, is_equivalent=True),
        PropositionEquivalenceDecision(position=4, is_equivalent=False),
        PropositionEquivalenceDecision(position=5, is_equivalent=False),
    ]
    local_definition_request = PropositionEquivalenceInput(
        proposition_content=(
            'In exercise A, f is defined by f(x)=x^2, so f(2)=4.'
        ),
        candidates=[
            PropositionIdentityCandidate(
                position=0,
                content=(
                    'In exercise B, f is defined by f(x)=sin(x), so '
                    'f(2)=sin(2).'
                ),
            )
        ],
    )
    local_definition_decision = [
        PropositionEquivalenceDecision(position=0, is_equivalent=False)
    ]
    return [
        dspy.Example(request=request, decisions=decisions).with_inputs(
            'request'
        ),
        dspy.Example(
            request=local_definition_request,
            decisions=local_definition_decision,
        ).with_inputs('request'),
    ]
