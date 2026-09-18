import asyncio

import pytest

from kms2.core.model import (
    SourceEntityHub,
    SourceEventHub,
    SourcePredicateHub,
    SourceProcedureHub,
    SourceStatementHub,
    SourceTripletHub,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms2.node.semantic.source_event_hub_persistence import (
    SourceEventHubPersistenceNode,
)
from kms2.node.semantic.source_predicate_hub_persistence import (
    SourcePredicateHubPersistenceNode,
)
from kms2.node.semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from kms2.node.semantic.source_triplet_hub_persistence import (
    SourceTripletHubPersistenceNode,
)


class _Repository:
    def __init__(self):
        self.calls = []

    async def replace_source_entity_hubs(self, *args):
        self.calls.append(('entity', args))

    async def replace_source_event_hubs(self, *args):
        self.calls.append(('event', args))

    async def replace_source_predicate_hubs(self, *args):
        self.calls.append(('predicate', args))

    async def replace_source_statement_hubs(self, *args):
        self.calls.append(('statement', args))

    async def replace_source_procedure_hubs(self, *args):
        self.calls.append(('procedure', args))

    async def replace_source_triplet_hubs(self, *args):
        self.calls.append(('triplet', args))


def _state(kind, hub):
    return SemanticState(
        source_uuid='source-1',
        **{
            f'source_{kind}_hubs': [hub],
            f'source_{kind}_hub_memberships': [[f'{kind}-1']],
        },
    )


@pytest.mark.parametrize(
    ('kind', 'node', 'hub'),
    [
        (
            'entity',
            SourceEntityHubPersistenceNode,
            SourceEntityHub(
                source_uuid='source-1',
                canonical_name='Alice',
                description='person',
                embedding=[0.1],
            ),
        ),
        (
            'event',
            SourceEventHubPersistenceNode,
            SourceEventHub(
                source_uuid='source-1',
                name='integration',
                description='process',
                embedding=[0.2],
            ),
        ),
        (
            'predicate',
            SourcePredicateHubPersistenceNode,
            SourcePredicateHub(
                source_uuid='source-1',
                predicate='supports',
                description='relation',
                embedding=[0.3],
            ),
        ),
        (
            'statement',
            SourceStatementHubPersistenceNode,
            SourceStatementHub(
                source_uuid='source-1',
                canonical_name='claim',
                description='fact',
                embedding=[0.4],
            ),
        ),
        (
            'procedure',
            SourceProcedureHubPersistenceNode,
            SourceProcedureHub(
                source_uuid='source-1',
                canonical_name='method',
                description='steps',
                embedding=[0.5],
            ),
        ),
        (
            'triplet',
            SourceTripletHubPersistenceNode,
            SourceTripletHub(
                source_uuid='source-1',
                canonical_name='support',
                description='relation',
                embedding=[0.6],
                subject_hub_uuid='subject-hub',
                predicate_hub_uuid='predicate-hub',
                object_hub_uuid='object-hub',
            ),
        ),
    ],
)
def test_each_hub_persistence_node_writes_only_its_typed_result(
    kind, node, hub
):
    repository = _Repository()
    result = asyncio.run(node(repository).run(_state(kind, hub)))
    assert [call[0] for call in repository.calls] == [kind]
    assert result == {f'source_{kind}_hub_count': 1}
