import pytest

from kms.construction import local_entity_hubs, local_statement_hubs, name_hubs
from kms.core import models


def test_entity_adjudicator_encodes_structured_comparison():
    left = models.HubMentionInput(name='group')
    right = models.HubMentionInput(name='algebraic group')

    encoded = local_entity_hubs.EntityHubAdjudicator.encode(
        None, left, right, 'algebra'
    )

    comparison = encoded['comparison']
    assert isinstance(comparison, models.HubMentionComparisonInput)
    assert comparison.left == left
    assert comparison.right == right
    assert comparison.scope == 'algebra'


def test_statement_synthesizer_encodes_structured_evidence():
    encoded = local_statement_hubs.LocalStatementHubSynthesizer.encode(
        None, ['A claim.']
    )

    assert encoded == {'request': models.EvidenceInput(evidence=['A claim.'])}


def test_lexical_membership_rejects_invalid_decision():
    with pytest.raises(ValueError):
        name_hubs._LexicalMembership.model_validate(
            {'decision': 'Maybe'}
        )
