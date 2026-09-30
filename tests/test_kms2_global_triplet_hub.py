import asyncio

from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHubDefinition,
    GlobalTripletHubEvidence,
    GlobalTripletHubGroup,
    GlobalTripletHubRole,
)
from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.database.schema import SCHEMA_STATEMENTS, VECTOR_INDEX_NAMES
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_triplet_hub import GlobalTripletHubNode
from kms2.node.global_semantic.global_triplet_hub_persistence import (
    GlobalTripletHubPersistenceNode,
)
from kms2.node.global_semantic.global_triplet_projection import (
    GlobalTripletProjectionNode,
)


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


class _Result:
    def __init__(self, rows=None):
        self._rows = rows or []

    async def data(self):
        return self._rows

    async def consume(self):
        return None


class _Session:
    def __init__(self, rows):
        self.rows = rows

    async def run(self, query, **parameters):
        return _Result(self.rows)


class _Repository:
    def __init__(self, groups):
        self.groups = groups
        self.replacements = []
        self.projection_count = 0

    async def read_global_triplet_hub_groups(self):
        return self.groups

    async def replace_global_triplet_hubs(self, hubs, memberships):
        self.replacements.append((hubs, memberships))

    async def replace_global_triplets(self):
        return self.projection_count


class _Module:
    def __init__(self):
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return GlobalTripletHubDefinition(
            canonical_name='supports relation',
            description='A cross-source support relation.',
        )


class _EmbeddingClient:
    def __init__(self):
        self.texts = []

    async def embed(self, texts):
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


def _group() -> GlobalTripletHubGroup:
    return GlobalTripletHubGroup(
        subject_hub_uuid='subject-global',
        subject_hub=GlobalTripletHubRole(name='Alice', description='A person.'),
        predicate_hub_uuid='predicate-global',
        predicate_hub=GlobalTripletHubRole(
            name='supports', description='A support relation.'
        ),
        object_hub_uuid='object-global',
        object_hub=GlobalTripletHubRole(
            name='Acme', description='An organization.'
        ),
        global_triplet_uuids=['global-b', 'global-a'],
        evidence=[
            GlobalTripletHubEvidence(
                global_triplet_uuid='global-b',
                source_triplet_hub_uuid='source-b',
                canonical_name='Supports',
                description='A support relation.',
            ),
            GlobalTripletHubEvidence(
                global_triplet_uuid='global-a',
                source_triplet_hub_uuid='source-a',
                canonical_name='Supports',
                description='A support relation.',
            ),
            GlobalTripletHubEvidence(
                global_triplet_uuid='global-a',
                source_triplet_hub_uuid='source-a',
                canonical_name='Supports',
                description='A support relation.',
            ),
        ],
    )


def test_global_triplet_repository_normalizes_group_order_and_memberships():
    row = {
        'subject_hub_uuid': 'subject-global',
        'subject_hub_name': 'Alice',
        'subject_hub_description': 'A person.',
        'predicate_hub_uuid': 'predicate-global',
        'predicate_hub_name': 'supports',
        'predicate_hub_description': 'A support relation.',
        'object_hub_uuid': 'object-global',
        'object_hub_name': 'Acme',
        'object_hub_description': 'An organization.',
        'global_triplet_uuids': ['global-b', 'global-a'],
        'evidence': [
            {
                'global_triplet_uuid': 'global-b',
                'source_triplet_hub_uuid': 'source-b',
                'canonical_name': 'Supports',
                'description': 'A support relation.',
            },
            {
                'global_triplet_uuid': 'global-a',
                'source_triplet_hub_uuid': 'source-a',
                'canonical_name': 'Supports',
                'description': 'A support relation.',
            },
        ],
    }
    session = _Session([row])
    repository = GlobalTripletRepository(lambda: _SessionContext(session))

    groups = asyncio.run(repository.read_global_triplet_hub_groups())

    assert groups[0].global_triplet_uuids == ['global-a', 'global-b']
    assert [item.source_triplet_hub_uuid for item in groups[0].evidence] == [
        'source-a',
        'source-b',
    ]


def test_global_triplet_hub_node_deduplicates_evidence_and_preserves_memberships():
    repository = _Repository([_group()])
    module = _Module()
    embedding = _EmbeddingClient()
    node = GlobalTripletHubNode(repository, module, embedding)

    async def run():
        state = GlobalSemanticState()
        state = state.model_copy(update=await node.load_groups(state))
        send = node.dispatch_synthesis(state)[0]
        results = await node.synthesis_worker(send.arg)
        state = state.model_copy(
            update={
                'global_triplet_hub_synthesis_results': results[
                    'global_triplet_hub_synthesis_results'
                ]
            }
        )
        state = state.model_copy(update=node.collect_synthesis(state))
        return state, await node.embed(state)

    state, result = asyncio.run(run())

    request = module.requests[0]
    assert request.source_triplet_hubs == ['Supports: A support relation.']
    assert request.global_triplets == ['Alice | supports | Acme']
    assert embedding.texts == [
        'supports relation: A cross-source support relation.'
    ]
    assert result['global_triplet_hub_memberships'] == [
        ['global-b', 'global-a']
    ]
    hub = result['global_triplet_hubs'][0]
    assert hub.subject_hub_uuid == 'subject-global'
    assert hub.predicate_hub_uuid == 'predicate-global'
    assert hub.object_hub_uuid == 'object-global'
    assert state.global_triplet_hub_synthesis_results_ordered[0].ordinal == 0


def test_global_triplet_hub_node_skips_empty_synthesis_and_embedding():
    repository = _Repository([])
    module = _Module()
    embedding = _EmbeddingClient()
    node = GlobalTripletHubNode(repository, module, embedding)

    async def run():
        state = GlobalSemanticState()
        state = state.model_copy(update=await node.load_groups(state))
        assert node.dispatch_synthesis(state) == (
            'global_triplet_hub_synthesis_collect'
        )
        state = state.model_copy(update=node.collect_synthesis(state))
        return await node.embed(state)

    assert asyncio.run(run()) == {
        'global_triplet_hubs': [],
        'global_triplet_hub_memberships': [],
    }
    assert module.requests == []
    assert embedding.texts == []


def test_global_triplet_projection_and_persistence_return_counts():
    repository = _Repository([])
    repository.projection_count = 2
    projection = GlobalTripletProjectionNode(repository)
    projection_result = asyncio.run(projection.run(GlobalSemanticState()))
    assert projection_result == {'global_triplet_count': 2}

    persistence = GlobalTripletHubPersistenceNode(repository)
    state = GlobalSemanticState(
        global_triplet_hubs=[], global_triplet_hub_memberships=[]
    )
    assert asyncio.run(persistence.run(state)) == {
        'global_triplet_hub_count': 0
    }
    assert repository.replacements == [([], [])]


def test_global_triplet_schema_registers_uuid_and_embedding():
    assert any(
        'global_triplet_uuid' in statement for statement in SCHEMA_STATEMENTS
    )
    assert any(
        'global_triplet_hub_uuid' in statement
        for statement in SCHEMA_STATEMENTS
    )
    assert 'global_triplet_hub_embedding' in VECTOR_INDEX_NAMES
