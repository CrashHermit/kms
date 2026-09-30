import asyncio

from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHub,
    SourceTripletHubDefinition,
    SourceTripletHubEvidence,
    SourceTripletHubGroup,
    SourceTripletHubRole,
)
from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.node.source_semantic.source_triplet_hub import SourceTripletHubNode
from kms2.node.source_semantic.source_triplet_hub_persistence import (
    SourceTripletHubPersistenceNode,
)


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    async def data(self):
        return self._rows

    async def consume(self):
        return None


class _RowsSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return _RowsResult(self.rows)


class _Repository:
    def __init__(self, groups):
        self.groups = groups
        self.replacements = []

    async def read_source_triplet_hub_groups(self, source_uuid):
        return self.groups

    async def replace_source_triplet_hubs(self, source_uuid, hubs, memberships):
        self.replacements.append((source_uuid, hubs, memberships))


class _Module:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return SourceTripletHubDefinition(
            canonical_name='supports relation',
            description='A source-grounded support relation.',
        )


class _EmbeddingClient:
    def __init__(self):
        self.texts = []

    async def embed(self, texts):
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


async def _run_triplet(node, state):
    state = state.model_copy(update=await node.load_groups(state))
    synthesis_results = []
    sends = node.dispatch_synthesis(state)
    if isinstance(sends, list):
        for send in sends:
            synthesis_results.extend(
                (await node.synthesis_worker(send.arg))[
                    'source_triplet_hub_synthesis_results'
                ]
            )
    state = state.model_copy(
        update={'source_triplet_hub_synthesis_results': synthesis_results}
    )
    state = state.model_copy(update=node.collect_synthesis(state))
    return await node.embed(state)


def _group() -> SourceTripletHubGroup:
    return SourceTripletHubGroup(
        subject_hub_uuid='subject-hub',
        subject_hub=SourceTripletHubRole(name='Alice', description='A person.'),
        predicate_hub_uuid='predicate-hub',
        predicate_hub=SourceTripletHubRole(
            name='supports', description='A support relation.'
        ),
        object_hub_uuid='object-hub',
        object_hub=SourceTripletHubRole(
            name='Acme', description='An organization.'
        ),
        triplet_uuids=['triplet-b', 'triplet-a'],
        evidence=[
            SourceTripletHubEvidence(
                triplet_uuid='triplet-b',
                fact_text='Alice supports Acme',
                subject='Alice',
                predicate='supports',
                object='Acme',
            ),
            SourceTripletHubEvidence(
                triplet_uuid='triplet-a',
                fact_text='Alice supports Acme',
                subject='Alice',
                predicate='supports',
                object='Acme',
            ),
            SourceTripletHubEvidence(
                triplet_uuid='triplet-c',
                fact_text='Acme supports Alice conditionally',
                subject='Acme',
                predicate='supports',
                object='Alice',
            ),
        ],
    )


def test_repository_maps_source_triplet_hub_groups():
    row = {
        'subject_hub_uuid': 'subject-hub',
        'subject_hub_name': 'Alice',
        'subject_hub_description': 'A person.',
        'predicate_hub_uuid': 'predicate-hub',
        'predicate_hub_name': 'supports',
        'predicate_hub_description': 'A support relation.',
        'object_hub_uuid': 'object-hub',
        'object_hub_name': 'Acme',
        'object_hub_description': 'An organization.',
        'triplet_uuids': ['triplet-a'],
        'evidence': [
            {
                'triplet_uuid': 'triplet-a',
                'fact_text': 'Alice supports Acme',
                'subject': 'Alice',
                'predicate': 'supports',
                'object': 'Acme',
            }
        ],
    }
    session = _RowsSession([row])
    repository = SourceTripletRepository(lambda: _SessionContext(session))

    groups = asyncio.run(repository.read_source_triplet_hub_groups('source-1'))

    assert groups == [
        SourceTripletHubGroup(
            subject_hub_uuid='subject-hub',
            subject_hub=SourceTripletHubRole(
                name='Alice', description='A person.'
            ),
            predicate_hub_uuid='predicate-hub',
            predicate_hub=SourceTripletHubRole(
                name='supports', description='A support relation.'
            ),
            object_hub_uuid='object-hub',
            object_hub=SourceTripletHubRole(
                name='Acme', description='An organization.'
            ),
            triplet_uuids=['triplet-a'],
            evidence=[
                SourceTripletHubEvidence(
                    triplet_uuid='triplet-a',
                    fact_text='Alice supports Acme',
                    subject='Alice',
                    predicate='supports',
                    object='Acme',
                )
            ],
        )
    ]


def test_source_triplet_node_deduplicates_model_evidence_and_retains_ids():
    repository = _Repository([_group()])
    module = _Module()
    embedding_client = _EmbeddingClient()
    node = SourceTripletHubNode(
        repository,
        module,
        embedding_client,
    )

    result = asyncio.run(
        _run_triplet(node, SourceSemanticState(source_uuid='source-1'))
    )

    request = module.requests[0]
    assert request.source_facts == [
        'Acme supports Alice conditionally',
        'Alice supports Acme',
    ]
    assert request.triplets == [
        'Acme | supports | Alice',
        'Alice | supports | Acme',
    ]
    assert embedding_client.texts == [
        'supports relation: A source-grounded support relation.'
    ]
    hub = result['source_triplet_hubs'][0]
    assert hub.source_uuid == 'source-1'
    assert hub.subject_hub_uuid == 'subject-hub'
    assert hub.predicate_hub_uuid == 'predicate-hub'
    assert hub.object_hub_uuid == 'object-hub'
    assert result['source_triplet_hub_memberships'] == [
        ['triplet-b', 'triplet-a']
    ]


def test_source_triplet_node_skips_synthesis_and_embedding_for_empty_groups():
    repository = _Repository([])
    module = _Module()
    embedding_client = _EmbeddingClient()
    node = SourceTripletHubNode(
        repository,
        module,
        embedding_client,
    )

    result = asyncio.run(
        _run_triplet(node, SourceSemanticState(source_uuid='source-1'))
    )

    assert result == {
        'source_triplet_hubs': [],
        'source_triplet_hub_memberships': [],
    }
    assert module.requests == []
    assert embedding_client.texts == []


def test_source_triplet_persistence_replaces_roles_and_raw_evidence():
    hub = SourceTripletHub(
        source_uuid='source-1',
        canonical_name='supports relation',
        description='A source-grounded support relation.',
        embedding=[0.1, 0.2],
        subject_hub_uuid='subject-hub',
        predicate_hub_uuid='predicate-hub',
        object_hub_uuid='object-hub',
    )
    repository = _Repository([])
    node = SourceTripletHubPersistenceNode(repository)

    result = asyncio.run(
        node.run(
            SourceSemanticState(
                source_uuid='source-1',
                source_triplet_hubs=[hub],
                source_triplet_hub_memberships=[['triplet-a', 'triplet-b']],
            )
        )
    )

    assert result == {'source_triplet_hub_count': 1}
    assert repository.replacements == [
        ('source-1', [hub], [['triplet-a', 'triplet-b']])
    ]


def test_repository_serializes_triplet_memberships_for_replacement():
    hub = SourceTripletHub(
        source_uuid='source-1',
        canonical_name='supports relation',
        description='A source-grounded support relation.',
        embedding=[0.1, 0.2],
        subject_hub_uuid='subject-hub',
        predicate_hub_uuid='predicate-hub',
        object_hub_uuid='object-hub',
    )
    session = _RowsSession([])
    repository = SourceTripletRepository(lambda: _SessionContext(session))

    asyncio.run(
        repository.replace_source_triplet_hubs(
            'source-1',
            [hub],
            [['triplet-a', 'triplet-b']],
        )
    )

    _, parameters = session.calls[0]
    assert parameters == {
        'source_uuid': 'source-1',
        'hubs': [
            {
                **hub.model_dump(),
                'triplet_uuids': ['triplet-a', 'triplet-b'],
            }
        ],
    }
