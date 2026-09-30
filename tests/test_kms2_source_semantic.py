import asyncio

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.source_semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionInput,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
    SourceEntityDescriptionTarget,
)
from kms2.core.model.source_semantic.source_event import SourceEvent
from kms2.core.model.source_semantic.source_fact_extraction import (
    SourceAtomicFact,
    SourceFact,
    SourceFactContext,
    SourceFactExtractionRequest,
    SourceFactTarget,
)
from kms2.core.model.source_semantic.source_predicate import SourcePredicate
from kms2.core.model.source_semantic.source_triplet import (
    SourceTriplet,
    SourceTripletOccurrence,
)
from kms2.core.model.source_semantic.source_triplet_decomposition import (
    SourceTripletDecompositionCandidate,
    SourceTripletDecompositionResult,
    SourceTripletEndpointKind,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source_semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source_semantic.source_event_repository import (
    SourceEventRepository,
)
from kms2.database.source_semantic.source_fact_repository import (
    SourceFactRepository,
)
from kms2.database.source_semantic.source_predicate_repository import (
    SourcePredicateRepository,
)
from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.graph import SourceSemanticGraph
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.node.source_semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.source_semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.source_semantic.source_entity_embedding import (
    SourceEntityEmbeddingNode,
)
from kms2.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms2.node.source_semantic.source_event_description import (
    SourceEventDescriptionNode,
)
from kms2.node.source_semantic.source_event_description_load import (
    SourceEventDescriptionLoadNode,
)
from kms2.node.source_semantic.source_event_embedding import (
    SourceEventEmbeddingNode,
)
from kms2.node.source_semantic.source_event_persistence import (
    SourceEventPersistenceNode,
)
from kms2.node.source_semantic.source_fact_extraction import (
    SourceFactExtractionNode,
)
from kms2.node.source_semantic.source_fact_persistence import (
    SourceFactPersistenceNode,
)
from kms2.node.source_semantic.source_predicate_description import (
    SourcePredicateDescriptionNode,
)
from kms2.node.source_semantic.source_predicate_description_load import (
    SourcePredicateDescriptionLoadNode,
)
from kms2.node.source_semantic.source_predicate_embedding import (
    SourcePredicateEmbeddingNode,
)
from kms2.node.source_semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from kms2.node.source_semantic.source_procedure_description import (
    SourceProcedureDescriptionNode,
)
from kms2.node.source_semantic.source_procedure_description_load import (
    SourceProcedureDescriptionLoadNode,
)
from kms2.node.source_semantic.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.source_semantic.source_procedure_persistence import (
    SourceProcedurePersistenceNode,
)
from kms2.node.source_semantic.source_statement_description import (
    SourceStatementDescriptionNode,
)
from kms2.node.source_semantic.source_statement_description_load import (
    SourceStatementDescriptionLoadNode,
)
from kms2.node.source_semantic.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.source_semantic.source_statement_persistence import (
    SourceStatementPersistenceNode,
)
from kms2.node.source_semantic.source_triplet_decomposition import (
    SourceTripletDecompositionNode,
)
from kms2.node.source_semantic.source_triplet_fact_load import (
    SourceTripletFactLoadNode,
)
from kms2.node.source_semantic.source_triplet_load import (
    SourceFactSourceLoadNode,
)
from kms2.node.source_semantic.source_triplet_persistence import (
    SourceTripletPersistenceNode,
)


def _block(
    uuid: str,
    content: str | None,
    block_type: str = 'paragraph',
) -> SourceBlock:
    return SourceBlock(uuid=uuid, block_type=block_type, content=content)


def test_fact_request_projects_uuid_free_pointer_context():
    blocks = [
        _block('before', 'before text'),
        _block('target', 'target text'),
        _block('after', 'after text'),
    ]
    request = SourceFactExtractionRequest(
        source_uuid='source-1',
        target=SourceFactTarget(uuid='target-1', source_blocks=[blocks[1]]),
        context_before=SourceFactContext(
            uuid='before-1',
            source_blocks=[blocks[0]],
        ),
        context_after=SourceFactContext(
            uuid='after-1',
            source_blocks=[blocks[2]],
        ),
    )

    model_input = request.model_input()

    assert model_input.model_dump() == {
        'context_before': [
            {'block_type': 'paragraph', 'content': 'before text'}
        ],
        'target_blocks': [
            {'block_type': 'paragraph', 'content': 'target text'}
        ],
        'context_after': [{'block_type': 'paragraph', 'content': 'after text'}],
    }
    assert 'uuid' not in model_input.model_dump_json()


class _FactExtractor:
    async def aforward(self, *, request):
        assert request.target_blocks[0].content == 'target text'
        return [SourceAtomicFact(text='Alice works for Acme')]


class _TripletDecomposer:
    async def aforward(self, *, request):
        return [
            SourceTripletDecompositionCandidate(
                subject='Alice',
                predicate='works for',
                object='Acme',
                subject_kind=SourceTripletEndpointKind.ENTITY,
                object_kind=SourceTripletEndpointKind.ENTITY,
            )
        ]


def test_fact_node_collects_pointer_facts_in_source_order():
    node = SourceFactExtractionNode(
        _FactExtractor(),
        ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
            target_budget=400,
        ),
    )
    state = SourceSemanticState(
        source_uuid='source-1',
        blocks=[_block('first', 'first text'), _block('second', 'target text')],
    )

    sends = node.dispatch(state)
    result = asyncio.run(node.worker(sends[1].arg))
    state.fact_results = result['fact_results']
    collected = node.collect(state)

    fact = collected['source_facts'][0]
    assert fact.text == 'Alice works for Acme'
    assert [block.uuid for block in fact.target.source_blocks] == ['second']
    assert fact.context_before.source_blocks[0].uuid == 'first'


def test_fact_dispatch_covers_every_persisted_block_in_order():
    node = SourceFactExtractionNode(
        _FactExtractor(),
        ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
            target_budget=400,
        ),
    )
    state = SourceSemanticState(
        source_uuid='source-1',
        blocks=[
            _block('paragraph', 'paragraph text'),
            _block('header', 'A heading', 'header'),
            _block('image', '', 'image'),
            _block('table', 'table text', 'table'),
            _block('empty', ''),
        ],
    )

    sends = node.dispatch(state)

    assert [
        send.arg['fact_extraction_request'].target.source_blocks[0].uuid
        for send in sends
    ] == ['paragraph', 'header', 'image', 'table', 'empty']


def _fact(uuid: str, block: SourceBlock) -> SourceFact:
    return SourceFact(
        uuid=uuid,
        text='Alice works for Acme',
        target=SourceFactTarget(source_blocks=[block]),
        context_before=SourceFactContext(source_blocks=[]),
        context_after=SourceFactContext(source_blocks=[]),
    )


def test_triplet_collection_reuses_each_persisted_fact():
    node = SourceTripletDecompositionNode(_TripletDecomposer())
    fact = _fact('fact-1', _block('block-1', 'source text'))
    state = SourceSemanticState(
        source_uuid='source-1',
        source_facts=[fact],
        triplet_results=[
            SourceTripletDecompositionResult(
                fact=fact,
                triplets=[
                    SourceTripletDecompositionCandidate(
                        subject='Alice',
                        predicate='works for',
                        object='Acme',
                        subject_kind=SourceTripletEndpointKind.ENTITY,
                        object_kind=SourceTripletEndpointKind.ENTITY,
                    ),
                    SourceTripletDecompositionCandidate(
                        subject='Alice',
                        predicate='works for',
                        object='Acme',
                        subject_kind=SourceTripletEndpointKind.ENTITY,
                        object_kind=SourceTripletEndpointKind.ENTITY,
                    ),
                ],
            )
        ],
    )

    collected = node.collect(state)
    triplet_occurrences = collected['triplet_occurrences']

    assert len(triplet_occurrences) == 2
    assert (
        triplet_occurrences[0].triplet.uuid
        != triplet_occurrences[1].triplet.uuid
    )
    assert all(occurrence.fact is fact for occurrence in triplet_occurrences)
    assert all(
        occurrence.subject.source_block_uuid == 'block-1'
        for occurrence in triplet_occurrences
    )


class _Result:
    async def consume(self):
        return None

    async def data(self):
        return [
            {'uuid': 'block-1', 'block_type': 'paragraph', 'content': 'text'}
        ]


class _Session:
    def __init__(self):
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return _Result()


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return None


def test_source_repository_loads_ordered_blocks():
    session = _Session()
    repository = SourceBlockRepository(lambda: _SessionContext(session))

    blocks = asyncio.run(repository.load_blocks('source-1'))

    assert [block.uuid for block in blocks] == ['block-1']
    assert blocks[0].content == 'text'


def test_fact_repository_serializes_only_node_identity_and_evidence_edges():
    session = _Session()
    repository = SourceFactRepository(lambda: _SessionContext(session))
    block = _block('block-1', 'source text')
    source_fact = _fact('fact-1', block)

    asyncio.run(repository.replace_source_facts('source-1', [source_fact]))

    _, parameters = session.calls[-1]
    assert parameters['facts'] == [
        {
            'uuid': 'fact-1',
            'text': 'Alice works for Acme',
            'target_uuid': source_fact.target.uuid,
            'context_before_uuid': source_fact.context_before.uuid,
            'context_after_uuid': source_fact.context_after.uuid,
            'target_block_uuids': ['block-1'],
            'context_before_block_uuids': [],
            'context_after_block_uuids': [],
        }
    ]


def test_triplet_repository_serializes_role_ids_only_as_match_parameters():
    session = _Session()
    repository = SourceTripletRepository(lambda: _SessionContext(session))
    block = _block('block-1', 'source text')
    source_fact = _fact('fact-1', block)
    occurrence = SourceTripletOccurrence(
        fact=source_fact,
        triplet=SourceTriplet(uuid='triplet-1'),
        subject=SourceEntity(
            uuid='subject-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Alice',
        ),
        object=SourceEntity(
            uuid='object-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Acme',
        ),
        predicate=SourcePredicate(
            uuid='predicate-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            predicate='works for',
        ),
    )

    asyncio.run(repository.replace_source_triplets('source-1', [occurrence]))

    _, parameters = session.calls[-1]
    assert parameters['triplets'] == [
        {
            'uuid': 'triplet-1',
            'subject_uuid': 'subject-1',
            'object_uuid': 'object-1',
            'predicate_uuid': 'predicate-1',
        }
    ]


class _Runtime:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _Database:
    def __init__(self, settings):
        self.closed = False

    async def close(self):
        self.closed = True


class _TypedSourceRepository:
    def __init__(self, blocks):
        self.blocks = blocks

    async def load_blocks(self, source_uuid):
        return self.blocks


class _TypedSemanticRepository:
    def __init__(self, entities, events=None, predicates=None):
        self.entities = entities
        self.events = events or []
        self.predicates = predicates or []
        self.source_facts = []
        self.triplet_occurrences = []
        self.updated = {}

    async def load_source_entities(self, source_uuid):
        return self.entities

    async def load_source_events(self, source_uuid):
        return self.events

    async def load_source_predicates(self, source_uuid):
        return self.predicates

    async def update_source_entity_description(self, source_uuid, results):
        self.updated['entity'] = (source_uuid, results)

    async def update_source_event_description(self, source_uuid, results):
        self.updated['event'] = (source_uuid, results)

    async def update_source_predicate_description(self, source_uuid, results):
        self.updated['predicate'] = (source_uuid, results)

    async def replace_source_facts(self, source_uuid, source_facts):
        self.source_facts = source_facts

    async def load_source_facts(self, source_uuid):
        return self.source_facts

    async def replace_source_triplets(self, source_uuid, triplet_occurrences):
        self.triplet_occurrences = triplet_occurrences

    async def load_source_statements(self, source_uuid):
        return []

    async def load_source_procedures(self, source_uuid):
        return []

    async def update_source_statement_description(self, source_uuid, results):
        self.updated['statement'] = (source_uuid, results)

    async def update_source_procedure_description(self, source_uuid, results):
        self.updated['procedure'] = (source_uuid, results)


class _EntityDescriber:
    async def aforward(self, *, request):
        return f'description of {request.term}'


class _EmbeddingClient:
    def __init__(self):
        self.texts = None

    async def embed(self, texts):
        self.texts = list(texts)
        return [[float(index)] for index, _ in enumerate(texts)]


class _RecordingNode:
    def __init__(self, node, marker, events):
        self.node = node
        self.marker = marker
        self.events = events

    def __getattr__(self, name):
        return getattr(self.node, name)

    async def run(self, state):
        self.events.append(self.marker)
        return await self.node.run(state)


class _HubNode:
    def __init__(self, key, events=None, event_name=None, requires=()):
        self.key = key
        self.events = events
        self.event_name = event_name or key
        self.requires = requires
        self.kind = key.removesuffix('_count')

    def _record(self, marker):
        if self.events is not None:
            assert all(
                requirement in self.events for requirement in self.requires
            )
            self.events.append(marker)

    async def load_candidates(self, state):
        self._record(f'{self.kind}_candidate_load')
        return {f'{self.kind}_candidates': []}

    def dispatch_rerank(self, state):
        return f'{self.kind}_rerank_collect'

    async def rerank_worker(self, state):
        return {}

    async def judge_worker(self, state):
        return {}

    def collect_rerank(self, state):
        self._record(f'{self.kind}_rerank_collect')
        return {}

    def dispatch_judge(self, state):
        return f'{self.kind}_judge_collect'

    def collect_judge(self, state):
        self._record(f'{self.kind}_judge_collect')
        return {}

    async def detect_communities(self, state):
        self._record(f'{self.kind}_communities')
        return {f'{self.kind}_communities': []}

    def dispatch_synthesis(self, state):
        return f'{self.kind}_synthesis_collect'

    def collect_synthesis(self, state):
        self._record(f'{self.kind}_synthesis_collect')
        return {}

    async def embed(self, state):
        self._record(f'{self.kind}_embedding')
        if self.kind == 'source_triplet_hub':
            return {
                'source_triplet_hubs': [],
                'source_triplet_hub_memberships': [],
            }
        kind = self.kind.removeprefix('source_').removesuffix('_hub')
        return {
            f'source_{kind}_hubs': [],
            f'source_{kind}_hub_memberships': [],
        }

    async def load_groups(self, state):
        self._record(self.event_name)
        return {'source_triplet_hub_groups': []}

    async def synthesis_worker(self, state):
        return {}

    async def run(self, state):
        self._record(self.event_name)
        return {self.key: 1}


class _FinalPersistence:
    async def run(self, state):
        return {
            'source_entity_hub_count': 1,
            'source_event_hub_count': 1,
            'source_predicate_hub_count': 1,
        }


def test_entity_load_orders_occurrences_and_projects_authoritative_context():
    blocks = [
        _block('block-1', 'first'),
        _block('block-2', 'second'),
    ]
    occurrences = [
        SourceEntity(
            uuid='entity-b',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Beta',
        ),
        SourceEntity(
            uuid='entity-a',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Alpha',
        ),
    ]
    node = SourceEntityDescriptionLoadNode(
        _TypedSourceRepository(blocks),
        _TypedSemanticRepository(occurrences),
        ContextWindowSettings(backward_budget=400, forward_budget=400),
    )

    state = asyncio.run(node.run(SourceSemanticState(source_uuid='source-1')))

    requests = state['source_entity_description_requests']
    assert [request.target.uuid for request in requests] == [
        'entity-a',
        'entity-b',
    ]
    assert requests[0].model_input.term == 'Alpha'
    assert requests[0].model_input.target_block.content == 'first'
    assert requests[0].model_input.context_before == []
    assert len(requests[0].model_input.context_after) == 1
    assert requests[0].model_input.context_after[0].content == 'second'


def test_entity_description_embedding_preserves_occurrence_identity():
    request = SourceEntityDescriptionRequest(
        target=SourceEntityDescriptionTarget(
            uuid='entity-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Alpha',
        ),
        model_input=SourceEntityDescriptionInput(
            term='Alpha',
            target_block={
                'block_type': 'paragraph',
                'content': 'Alpha appears here',
            },
        ),
    )
    describer = SourceEntityDescriptionNode(_EntityDescriber())
    description_state = asyncio.run(
        describer.worker({'source_entity_description_request': request})
    )
    description_result = description_state['source_entity_description_results'][
        0
    ]
    client = _EmbeddingClient()
    embedding = SourceEntityEmbeddingNode(client)

    embedded_state = asyncio.run(
        embedding.worker(
            {'source_entity_description_results': [description_result]}
        )
    )

    assert client.texts == ['Alpha : description of Alpha']
    result = embedded_state['source_entity_embedding_results'][0]
    assert result.uuid == 'entity-1'
    assert result.embedding == [0.0]


class _RowsResult:
    async def data(self):
        return [
            {
                'uuid': 'entity-1',
                'source_uuid': 'source-1',
                'source_block_uuid': 'block-1',
                'name': 'Alpha',
                'predicate': 'supports',
                'description': 'supports locally',
                'score': 0.99,
            },
            {
                'uuid': 'entity-2',
                'source_uuid': 'source-1',
                'source_block_uuid': 'block-2',
                'name': 'Beta',
                'predicate': 'contains',
                'description': 'contains locally',
                'score': 0.81,
            },
        ]

    async def consume(self):
        return None


class _RowsSession:
    def __init__(self):
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return _RowsResult()


def test_semantic_repository_maps_typed_reads_and_updates():
    session = _RowsSession()
    entity_repository = SourceEntityRepository(lambda: _SessionContext(session))
    event_repository = SourceEventRepository(lambda: _SessionContext(session))
    predicate_repository = SourcePredicateRepository(
        lambda: _SessionContext(session)
    )

    entities = asyncio.run(entity_repository.load_source_entities('source-1'))
    result = SourceEntityDescriptionResult(
        **entities[0].model_dump(),
        description='a local description',
        embedding=[0.1, 0.2],
    )
    asyncio.run(
        entity_repository.update_source_entity_description('source-1', [result])
    )

    assert session.calls[1][1]['rows'] == [
        {
            'uuid': 'entity-1',
            'description': 'a local description',
            'embedding': [0.1, 0.2],
        }
    ]
    entity_matches = asyncio.run(
        entity_repository.find_similar_source_entities('entity-1', top_k=2)
    )
    event_matches = asyncio.run(
        event_repository.find_similar_source_events('event-1', top_k=2)
    )
    predicate_matches = asyncio.run(
        predicate_repository.find_similar_source_predicates(
            'predicate-1', top_k=2
        )
    )

    assert [match.uuid for match in entity_matches] == ['entity-1', 'entity-2']
    assert [match.uuid for match in event_matches] == ['entity-1', 'entity-2']
    assert [match.uuid for match in predicate_matches] == [
        'entity-1',
        'entity-2',
    ]
    assert entity_matches[0].score == 0.99
    assert entity_matches[0].model_dump().keys() == {
        'uuid',
        'source_uuid',
        'source_block_uuid',
        'name',
        'description',
        'score',
    }
    assert predicate_matches[0].predicate == 'supports'


def test_semantic_graph_runs_all_typed_phases_in_one_graph():
    source_repository = _TypedSourceRepository(
        [_block('block-1', 'target text')]
    )
    semantic_repository = _TypedSemanticRepository(
        [
            SourceEntity(
                uuid='entity-1',
                source_uuid='source-1',
                source_block_uuid='block-1',
                name='Alpha',
            )
        ],
        [
            SourceEvent(
                uuid='event-1',
                source_uuid='source-1',
                source_block_uuid='block-1',
                name='Appears',
            )
        ],
        [
            SourcePredicate(
                uuid='predicate-1',
                source_uuid='source-1',
                source_block_uuid='block-1',
                predicate='supports',
            )
        ],
    )
    context_window = ContextWindowSettings(
        backward_budget=400,
        forward_budget=400,
        target_budget=400,
    )

    events = []
    graph = SourceSemanticGraph(
        source_fact_source_load=SourceFactSourceLoadNode(source_repository),
        source_fact_extraction=SourceFactExtractionNode(
            _FactExtractor(), context_window
        ),
        source_fact_persistence=SourceFactPersistenceNode(semantic_repository),
        source_triplet_fact_load=SourceTripletFactLoadNode(semantic_repository),
        source_triplet_decomposition=SourceTripletDecompositionNode(
            _TripletDecomposer()
        ),
        source_triplet_persistence=SourceTripletPersistenceNode(
            semantic_repository
        ),
        source_entity_description_load=_RecordingNode(
            SourceEntityDescriptionLoadNode(
                source_repository,
                semantic_repository,
                context_window,
            ),
            'entity_description_load',
            events,
        ),
        source_entity_description=SourceEntityDescriptionNode(
            _EntityDescriber()
        ),
        source_entity_embedding=SourceEntityEmbeddingNode(_EmbeddingClient()),
        source_entity_persistence=_RecordingNode(
            SourceEntityPersistenceNode(semantic_repository),
            'entity_description_persist',
            events,
        ),
        source_event_description_load=_RecordingNode(
            SourceEventDescriptionLoadNode(
                source_repository,
                semantic_repository,
                context_window,
            ),
            'event_description_load',
            events,
        ),
        source_event_description=SourceEventDescriptionNode(_EntityDescriber()),
        source_event_embedding=SourceEventEmbeddingNode(_EmbeddingClient()),
        source_event_persistence=_RecordingNode(
            SourceEventPersistenceNode(semantic_repository),
            'event_description_persist',
            events,
        ),
        source_predicate_description_load=_RecordingNode(
            SourcePredicateDescriptionLoadNode(
                source_repository,
                semantic_repository,
                context_window,
            ),
            'predicate_description_load',
            events,
        ),
        source_predicate_description=SourcePredicateDescriptionNode(
            _EntityDescriber()
        ),
        source_predicate_embedding=SourcePredicateEmbeddingNode(
            _EmbeddingClient()
        ),
        source_predicate_persistence=_RecordingNode(
            SourcePredicatePersistenceNode(semantic_repository),
            'predicate_description_persist',
            events,
        ),
        source_statement_description_load=_RecordingNode(
            SourceStatementDescriptionLoadNode(
                source_repository,
                semantic_repository,
                context_window,
            ),
            'statement_description_load',
            events,
        ),
        source_statement_description=SourceStatementDescriptionNode(
            _EntityDescriber()
        ),
        source_statement_embedding=SourceStatementEmbeddingNode(
            _EmbeddingClient()
        ),
        source_statement_persistence=_RecordingNode(
            SourceStatementPersistenceNode(semantic_repository),
            'statement_description_persist',
            events,
        ),
        source_procedure_description_load=_RecordingNode(
            SourceProcedureDescriptionLoadNode(
                source_repository,
                semantic_repository,
                context_window,
            ),
            'procedure_description_load',
            events,
        ),
        source_procedure_description=SourceProcedureDescriptionNode(
            _EntityDescriber()
        ),
        source_procedure_embedding=SourceProcedureEmbeddingNode(
            _EmbeddingClient()
        ),
        source_procedure_persistence=_RecordingNode(
            SourceProcedurePersistenceNode(semantic_repository),
            'procedure_description_persist',
            events,
        ),
        source_entity_hub=_HubNode(
            'source_entity_hub_count',
            events=events,
        ),
        source_event_hub=_HubNode('source_event_hub_count'),
        source_predicate_hub=_HubNode('source_predicate_hub_count'),
        source_triplet_hub=_HubNode(
            'source_triplet_hub_count',
            events=events,
            event_name='triplet_staged',
            requires=(
                'entity_persisted',
                'event_persisted',
                'predicate_persisted',
                'statement_persisted',
                'procedure_persisted',
            ),
        ),
        source_statement_hub=_HubNode('source_statement_hub_count'),
        source_procedure_hub=_HubNode('source_procedure_hub_count'),
        source_entity_hub_persistence=_HubNode(
            'source_entity_hub_count',
            events=events,
            event_name='entity_persisted',
        ),
        source_event_hub_persistence=_HubNode(
            'source_event_hub_count',
            events=events,
            event_name='event_persisted',
        ),
        source_predicate_hub_persistence=_HubNode(
            'source_predicate_hub_count',
            events=events,
            event_name='predicate_persisted',
        ),
        source_statement_hub_persistence=_HubNode(
            'source_statement_hub_count',
            events=events,
            event_name='statement_persisted',
        ),
        source_procedure_hub_persistence=_HubNode(
            'source_procedure_hub_count',
            events=events,
            event_name='procedure_persisted',
        ),
        source_triplet_hub_persistence=_HubNode(
            'source_triplet_hub_count',
            events=events,
            event_name='triplet_persisted',
        ),
    ).build_graph()
    assert {
        'source_triplet_persistence',
        'source_entity_description_load',
        'source_event_description_load',
        'source_predicate_description_load',
        'source_entity_hub_load',
        'source_event_hub_load',
        'source_predicate_hub_load',
        'source_statement_hub_load',
        'source_procedure_hub_load',
        'source_triplet_hub_load',
        'source_triplet_hub_persistence',
    } <= set(graph.nodes)
    final_state = asyncio.run(graph.ainvoke({'source_uuid': 'source-1'}))
    kinds = ('entity', 'event', 'predicate', 'statement', 'procedure')
    for previous, current in zip(kinds, kinds[1:], strict=False):
        assert events.index(f'{previous}_description_persist') < events.index(
            f'{current}_description_load'
        )
    persistence_positions = [
        events.index(f'{kind}_description_persist') for kind in kinds
    ]
    assert persistence_positions == sorted(persistence_positions)
    assert events.index('source_entity_hub_candidate_load') > events.index(
        'procedure_description_persist'
    )
    assert events.index('triplet_staged') > max(
        events.index('entity_persisted'),
        events.index('event_persisted'),
        events.index('predicate_persisted'),
        events.index('statement_persisted'),
        events.index('procedure_persisted'),
    )
    assert events.index('triplet_persisted') > events.index('triplet_staged')

    assert len(final_state['triplet_occurrences']) == 1
    assert len(final_state['source_facts']) == 1
    assert final_state['source_entity_description_persisted_count'] == 1
    assert final_state['source_event_description_persisted_count'] == 1
    assert final_state['source_predicate_description_persisted_count'] == 1
    assert set(semantic_repository.updated) == {'entity', 'event', 'predicate'}
    assert final_state['source_entity_hub_count'] == 1
    assert final_state['source_event_hub_count'] == 1
    assert final_state['source_predicate_hub_count'] == 1
    assert final_state['source_triplet_hub_count'] == 1
    assert final_state['source_statement_hub_count'] == 1
    assert final_state['source_procedure_hub_count'] == 1
