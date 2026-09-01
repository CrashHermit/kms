import pytest

from kms.construction import local_statement_hubs, name_hubs
from kms.core import models


def test_statement_synthesizer_encodes_structured_evidence():
    encoded = local_statement_hubs.StatementHubSynthesizer.encode(
        None,
        [],
        ['A claim.'],
        'source-local statements',
    )

    assert encoded == {
        'request': models.HubSynthesisInput(
            surface_forms=[],
            descriptions=['A claim.'],
            scope='source-local statements',
        )
    }


def test_lexical_membership_rejects_invalid_decision():
    with pytest.raises(ValueError):
        name_hubs._LexicalMembership.model_validate({'decision': 'Maybe'})
