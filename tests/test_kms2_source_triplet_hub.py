import asyncio
import json

from kms2_token_helpers import direct_synthesis

from kms2.core.model.source_semantic.source_triplet_context import (
    SourceTripletContextTarget,
)
from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHub,
    SourceTripletHubDefinition,
    SourceTripletHubEvidence,
    SourceTripletHubGroup,
    SourceTripletHubRole,
    SourceTripletHubSummary,
)
from kms2.core.windowing import TokenBudget
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

    async def acall(self, *, request):
        self.requests.append(request)
        return SourceTripletHubDefinition(
            canonical_name='supports relation',
            description='A source-grounded support relation.',
        )


class _ContentAwareCounter:
    def count_texts(self, texts):
        counts = []
        for text in texts:
            payload = json.loads(text)
            if 'evidence' in payload:
                counts.append(
                    sum(
                        1
                        + len(item['qualifiers'])
                        + len(item['interpretive_bindings'])
                        for item in payload['evidence']
                    )
                )
            elif 'triplet_evidence' in payload:
                item = payload['triplet_evidence']
                counts.append(
                    1
                    + len(item['qualifiers'])
                    + len(item['interpretive_bindings'])
                )
            elif 'summaries' in payload:
                counts.append(len(payload['summaries']))
            else:
                counts.append(1)
        return counts


class _SummaryModule:
    def __init__(self):
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        return SourceTripletHubSummary(
            text='The supplied records retain their scoped conditions and meanings.'
        )


class _MergeModule:
    def __init__(self):
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        return SourceTripletHubSummary(
            text='Each condition and meaning remains associated with its claim.'
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
        triplet_uuids=['triplet-b', 'triplet-a', 'triplet-c'],
        evidence=[
            SourceTripletHubEvidence(
                triplet_uuid='triplet-b',
                fact_text='Alice supports Acme',
                subject='Alice',
                predicate='supports',
                object='Acme',
                qualifiers=[
                    {
                        'target': SourceTripletContextTarget.TRIPLET,
                        'relation': 'condition',
                        'value': 'during business hours',
                    }
                ],
                interpretive_bindings=[
                    {
                        'target': SourceTripletContextTarget.SUBJECT,
                        'term': 'Alice',
                        'meaning': 'the researcher',
                    }
                ],
            ),
            SourceTripletHubEvidence(
                triplet_uuid='triplet-a',
                fact_text='Alice supports Acme',
                subject='Alice',
                predicate='supports',
                object='Acme',
                qualifiers=[
                    {
                        'target': SourceTripletContextTarget.TRIPLET,
                        'relation': 'condition',
                        'value': 'outside business hours',
                    }
                ],
                interpretive_bindings=[
                    {
                        'target': SourceTripletContextTarget.SUBJECT,
                        'term': 'Alice',
                        'meaning': 'the client',
                    }
                ],
            ),
            SourceTripletHubEvidence(
                triplet_uuid='triplet-c',
                fact_text='Acme supports Alice conditionally',
                subject='Acme',
                predicate='supports',
                object='Alice',
                qualifiers=[],
                interpretive_bindings=[],
            ),
        ],
    )


def test_source_triplet_synthesis_preserves_scoped_occurrence_evidence():
    repository = _Repository([_group()])
    module = _Module()
    embedding_client = _EmbeddingClient()
    node = SourceTripletHubNode(
        repository,
        module,
        embedding_client,
        **direct_synthesis(include_reranker=False),
    )

    result = asyncio.run(
        _run_triplet(node, SourceSemanticState(source_uuid='source-1'))
    )

    request = module.requests[0]
    assert [item.fact_text for item in request.evidence] == [
        'Alice supports Acme',
        'Alice supports Acme',
        'Acme supports Alice conditionally',
    ]
    assert [
        item.qualifiers[0].value if item.qualifiers else None
        for item in request.evidence
    ] == ['outside business hours', 'during business hours', None]
    assert [
        item.interpretive_bindings[0].meaning
        if item.interpretive_bindings
        else None
        for item in request.evidence
    ] == ['the client', 'the researcher', None]
    assert 'triplet_uuid' not in request.model_dump_json()
    assert embedding_client.texts == [
        'supports relation: A source-grounded support relation.'
    ]
    hub = result['source_triplet_hubs'][0]
    assert hub.source_uuid == 'source-1'
    assert hub.subject_hub_uuid == 'subject-hub'
    assert hub.predicate_hub_uuid == 'predicate-hub'
    assert hub.object_hub_uuid == 'object-hub'
    assert result['source_triplet_hub_memberships'] == [
        ['triplet-b', 'triplet-a', 'triplet-c']
    ]


def test_source_triplet_summary_preserves_complete_claim_records():
    repository = _Repository([_group()])
    module = _Module()
    summary_module = _SummaryModule()
    merge_module = _MergeModule()
    embedding_client = _EmbeddingClient()
    counter = _ContentAwareCounter()
    node = SourceTripletHubNode(
        repository,
        module,
        embedding_client,
        summary_module=summary_module,
        merge_module=merge_module,
        final_budget=TokenBudget(counter, 1),
        summary_budget=TokenBudget(counter, 5),
        merge_budget=TokenBudget(counter, 100),
    )

    result = asyncio.run(
        _run_triplet(node, SourceSemanticState(source_uuid='source-1'))
    )

    direct_weight = sum(
        1 + len(item.qualifiers) + len(item.interpretive_bindings)
        for item in _group().evidence
    )
    assert direct_weight > 1
    leaf_claims = [
        json.loads(item)['triplet_evidence']
        for request in summary_module.requests
        for item in request.evidence
        if 'triplet_evidence' in json.loads(item)
    ]
    assert len(leaf_claims) == 3
    assert sorted(
        claim['qualifiers'][0]['value']
        for claim in leaf_claims
        if claim['qualifiers']
    ) == ['during business hours', 'outside business hours']
    assert sorted(
        claim['interpretive_bindings'][0]['meaning']
        for claim in leaf_claims
        if claim['interpretive_bindings']
    ) == ['the client', 'the researcher']
    assert [claim for claim in leaf_claims if not claim['qualifiers']] == [
        {
            'fact_text': 'Acme supports Alice conditionally',
            'subject': 'Acme',
            'predicate': 'supports',
            'object': 'Alice',
            'qualifiers': [],
            'interpretive_bindings': [],
        }
    ]
    assert merge_module.requests
    assert len(module.requests[0].summaries) == 1
    assert result['source_triplet_hubs']


def test_source_triplet_node_skips_synthesis_and_embedding_for_empty_groups():
    repository = _Repository([])
    module = _Module()
    embedding_client = _EmbeddingClient()
    node = SourceTripletHubNode(
        repository,
        module,
        embedding_client,
        **direct_synthesis(include_reranker=False),
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
