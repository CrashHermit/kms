import asyncio

from kms2.config import ContextWindowSettings
from kms2.core.context_window import select_window
from kms2.core.model import (
    AtomicFact,
    Entity,
    EntityEnrichmentRequest,
    EntityEnrichmentResult,
    EntityEnrichmentTarget,
    Event,
    ExtractedFact,
    FactExtractionRequest,
    Predicate,
    RawAssertion,
    RawTriplet,
    SemanticNodeKind,
    SourceBlock,
    TermEnrichmentInput,
    TripletCandidate,
    TripletDecompositionResult,
)
from kms2.database.semantic.queries import (
    CLEAR_SOURCE_ASSERTIONS,
    READ_SOURCE_ENTITIES,
    REPLACE_SOURCE_ASSERTIONS,
    UPDATE_ENTITY_ENRICHMENT,
)
from kms2.database.semantic.repository import SemanticRepository
from kms2.database.source.queries import READ_SOURCE_BLOCKS
from kms2.database.source.repository import SourceRepository
from kms2.langgraph.semantic.graph import SemanticGraph
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.entity_embedding import EntityEmbeddingNode
from kms2.node.semantic.entity_enrichment import EntityEnrichmentNode
from kms2.node.semantic.entity_enrichment_load import EntityEnrichmentLoadNode
from kms2.node.semantic.entity_enrichment_persistence import (
    EntityEnrichmentPersistenceNode,
)
from kms2.node.semantic.event_embedding import EventEmbeddingNode
from kms2.node.semantic.event_enrichment import EventEnrichmentNode
from kms2.node.semantic.event_enrichment_load import EventEnrichmentLoadNode
from kms2.node.semantic.event_enrichment_persistence import (
    EventEnrichmentPersistenceNode,
)
from kms2.node.semantic.predicate_embedding import PredicateEmbeddingNode
from kms2.node.semantic.predicate_enrichment import PredicateEnrichmentNode
from kms2.node.semantic.predicate_enrichment_load import (
    PredicateEnrichmentLoadNode,
)
from kms2.node.semantic.predicate_enrichment_persistence import (
    PredicateEnrichmentPersistenceNode,
)
from kms2.node.semantic.triplet import (
    FactExtractionNode,
    TripletDecompositionNode,
)
from kms2.node.semantic.triplet_load import TripletSourceLoadNode
from kms2.node.semantic.triplet_persistence import TripletPersistenceNode


def _block(
    uuid: str,
    content: str | None,
    block_type: str = 'paragraph',
) -> SourceBlock:
    return SourceBlock(uuid=uuid, block_type=block_type, content=content)


def test_fact_request_projects_uuid_free_context_window():
    blocks = [
        _block('before', 'before text'),
        _block('target', 'target text'),
        _block('after', 'after text'),
    ]
    request = FactExtractionRequest(
        source_uuid='source-1',
        target_block=blocks[1],
        window=select_window(
            blocks,
            [1],
            backward_budget=400,
            forward_budget=400,
        ),
    )

    model_input = request.model_input()

    assert model_input.model_dump() == {
        'context_before': [
            {
                'block_type': 'paragraph',
                'content': 'before text',
                'asset_paths': [],
            }
        ],
        'target_block': {
            'block_type': 'paragraph',
            'content': 'target text',
            'asset_paths': [],
        },
        'context_after': [
            {
                'block_type': 'paragraph',
                'content': 'after text',
                'asset_paths': [],
            }
        ],
    }
    assert 'uuid' not in model_input.model_dump_json()


class _FactExtractor:
    async def aforward(self, *, request):
        assert request.target_block.content == 'target text'
        return [AtomicFact(text='Alice works for Acme')]


class _TripletDecomposer:
    async def aforward(self, *, request):
        return [
            TripletCandidate(
                subject='Alice',
                predicate='works for',
                object='Acme',
                subject_kind=SemanticNodeKind.ENTITY,
                object_kind=SemanticNodeKind.ENTITY,
            )
        ]


def test_fact_node_collects_results_in_source_order():
    node = FactExtractionNode(
        _FactExtractor(),
        ContextWindowSettings(backward_budget=400, forward_budget=400),
    )
    state = SemanticState(
        source_uuid='source-1',
        blocks=[_block('first', 'first text'), _block('second', 'target text')],
    )

    sends = node.dispatch(state)
    assert len(sends) == 2
    result = asyncio.run(node.worker(sends[1].arg))
    state.fact_results = result['fact_results']
    collected = node.collect(state)
    assert [
        fact.source_block_uuid for fact in collected['extracted_facts']
    ] == ['second']


def test_fact_dispatch_covers_every_persisted_block_in_order():
    node = FactExtractionNode(
        _FactExtractor(),
        ContextWindowSettings(backward_budget=400, forward_budget=400),
    )
    state = SemanticState(
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
        send.arg['fact_extraction_request'].target_block.uuid for send in sends
    ] == ['paragraph', 'header', 'image', 'table', 'empty']


def test_triplet_collection_creates_fresh_vertex_ids_and_preserves_provenance():
    node = TripletDecompositionNode(_TripletDecomposer())
    fact = ExtractedFact(
        source_uuid='source-1',
        source_block_uuid='block-1',
        text='Alice works for Acme',
    )
    state = SemanticState(
        source_uuid='source-1',
        extracted_facts=[fact],
        triplet_results=[
            TripletDecompositionResult(
                fact=fact,
                triplets=[
                    TripletCandidate(
                        subject='Alice',
                        predicate='works for',
                        object='Acme',
                        subject_kind=SemanticNodeKind.ENTITY,
                        object_kind=SemanticNodeKind.ENTITY,
                    ),
                    TripletCandidate(
                        subject='Alice',
                        predicate='works for',
                        object='Acme',
                        subject_kind=SemanticNodeKind.ENTITY,
                        object_kind=SemanticNodeKind.ENTITY,
                    ),
                ],
            )
        ],
    )

    assertions = node.collect(state)['raw_assertions']

    assert len(assertions) == 2
    assert assertions[0].triplet.uuid != assertions[1].triplet.uuid
    assert assertions[0].subject.uuid != assertions[1].subject.uuid
    assert all(
        assertion.triplet.source_block_uuid == 'block-1'
        for assertion in assertions
    )
    assert all(
        isinstance(assertion.subject, Entity) for assertion in assertions
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
    repository = SourceRepository(lambda: _SessionContext(session))

    blocks = asyncio.run(repository.load_blocks('source-1'))

    assert session.calls == [(READ_SOURCE_BLOCKS, {'source_uuid': 'source-1'})]
    assert [block.uuid for block in blocks] == ['block-1']
    assert blocks[0].content == 'text'


def test_semantic_repository_replace_uses_exact_query_parameters():
    session = _Session()
    repository = SemanticRepository(lambda: _SessionContext(session))
    assertion = RawAssertion(
        triplet=RawTriplet(
            source_uuid='source-1',
            source_block_uuid='block-1',
            subject_uuid='subject-1',
            object_uuid='object-1',
            predicate_uuid='predicate-1',
        ),
        subject=Entity(
            uuid='subject-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Alice',
        ),
        object=Entity(
            uuid='object-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Acme',
        ),
        predicate=Predicate(
            uuid='predicate-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            predicate='works for',
        ),
    )

    asyncio.run(repository.replace_source_assertions('source-1', [assertion]))
    assert session.calls[-1][0] is REPLACE_SOURCE_ASSERTIONS
    assert session.calls[-1][1]['source_uuid'] == 'source-1'
    assert session.calls[-1][1]['triplets'][0]['uuid'] == assertion.triplet.uuid
    query = session.calls[-1][0]
    assert 'CREATE (block)-[:HAS_TRIPLET]->(triplet)' in query
    assert 'CREATE (source)-[:HAS_TRIPLET]->(triplet)' not in query
    assert 'CREATE (triplet)-[:HAS_SUBJECT]->(subject)' in query
    assert 'CREATE (triplet)-[:HAS_OBJECT]->(object)' in query
    assert 'CREATE (triplet)-[:HAS_PREDICATE]->(predicate)' in query

    asyncio.run(repository.replace_source_assertions('source-1', []))
    assert session.calls[-1][0] is CLEAR_SOURCE_ASSERTIONS


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


class _ComposedSemanticGraph:
    def build_graph(self, raw_only=False):
        return self

    async def ainvoke(self, initial_state):
        assert initial_state == {'source_uuid': 'source-1'}
        return {'raw_assertions': [object(), object()]}


def test_extract_triplets_owns_runtime_and_returns_assertion_count(monkeypatch):
    from kms2 import application
    from kms2.config import Settings

    settings = Settings()
    database = _Database(settings.database)
    graph = _ComposedSemanticGraph()
    monkeypatch.setattr(application, 'LocalModelRuntime', lambda _: _Runtime())
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(application, 'build_semantic_graph', lambda *_: graph)

    count = asyncio.run(application.extract_triplets(settings, 'source-1'))

    assert count == 2
    assert database.closed is True


class _EmptySourceRepository:
    async def load_blocks(self, source_uuid):
        assert source_uuid == 'source-1'
        return []


class _EmptySemanticRepository:
    def __init__(self):
        self.assertions = None

    async def replace_source_assertions(self, source_uuid, assertions):
        self.assertions = (source_uuid, assertions)


async def _noop_schema():
    return None


def test_semantic_graph_raw_only_mode_cleans_semantic_layer():
    from kms2.node.semantic.triplet_load import TripletSourceLoadNode
    from kms2.node.semantic.triplet_persistence import TripletPersistenceNode

    repository = _EmptySemanticRepository()
    source_repository = _EmptySourceRepository()
    semantic_repository = _TypedSemanticRepository([])
    context_window = ContextWindowSettings(
        backward_budget=400,
        forward_budget=400,
    )
    graph = SemanticGraph(
        triplet_source_load=TripletSourceLoadNode(source_repository),
        fact_extraction=FactExtractionNode(_FactExtractor(), context_window),
        triplet_decomposition=TripletDecompositionNode(_TripletDecomposer()),
        triplet_persistence=TripletPersistenceNode(repository, _noop_schema),
        entity_enrichment_load=EntityEnrichmentLoadNode(
            source_repository, semantic_repository, context_window
        ),
        entity_enrichment=EntityEnrichmentNode(_EntityEnricher()),
        entity_embedding=EntityEmbeddingNode(_EmbeddingClient()),
        entity_enrichment_persistence=EntityEnrichmentPersistenceNode(
            semantic_repository, _noop_schema
        ),
        event_enrichment_load=EventEnrichmentLoadNode(
            source_repository, semantic_repository, context_window
        ),
        event_enrichment=EventEnrichmentNode(_EntityEnricher()),
        event_embedding=EventEmbeddingNode(_EmbeddingClient()),
        event_enrichment_persistence=EventEnrichmentPersistenceNode(
            semantic_repository, _noop_schema
        ),
        predicate_enrichment_load=PredicateEnrichmentLoadNode(
            source_repository, semantic_repository, context_window
        ),
        predicate_enrichment=PredicateEnrichmentNode(_EntityEnricher()),
        predicate_embedding=PredicateEmbeddingNode(_EmbeddingClient()),
        predicate_enrichment_persistence=PredicateEnrichmentPersistenceNode(
            semantic_repository, _noop_schema
        ),
    ).build_graph(raw_only=True)

    assert {'triplet_source_load', 'triplet_persistence'} <= set(graph.nodes)
    assert 'entity_enrichment_load' not in graph.nodes
    final_state = asyncio.run(graph.ainvoke({'source_uuid': 'source-1'}))

    assert final_state['raw_assertions'] == []
    assert repository.assertions == ('source-1', [])


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
        self.updated = {}

    async def load_entities(self, source_uuid):
        return self.entities

    async def load_events(self, source_uuid):
        return self.events

    async def load_predicates(self, source_uuid):
        return self.predicates

    async def update_entity_enrichment(self, source_uuid, results):
        self.updated['entity'] = (source_uuid, results)

    async def update_event_enrichment(self, source_uuid, results):
        self.updated['event'] = (source_uuid, results)

    async def update_predicate_enrichment(self, source_uuid, results):
        self.updated['predicate'] = (source_uuid, results)

    async def replace_source_assertions(self, source_uuid, assertions):
        self.assertions = (source_uuid, assertions)


class _EntityEnricher:
    async def aforward(self, *, request):
        return f'description of {request.term}'


class _EmbeddingClient:
    def __init__(self):
        self.texts = None

    async def embed(self, texts):
        self.texts = list(texts)
        return [[float(index)] for index, _ in enumerate(texts)]


def test_entity_load_orders_occurrences_and_projects_authoritative_context():
    blocks = [
        _block('block-1', 'first'),
        _block('block-2', 'second'),
    ]
    occurrences = [
        Entity(
            uuid='entity-b',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Beta',
        ),
        Entity(
            uuid='entity-a',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Alpha',
        ),
    ]
    node = EntityEnrichmentLoadNode(
        _TypedSourceRepository(blocks),
        _TypedSemanticRepository(occurrences),
        ContextWindowSettings(backward_budget=400, forward_budget=400),
    )

    state = asyncio.run(node.run(SemanticState(source_uuid='source-1')))

    requests = state['entity_requests']
    assert [request.target.uuid for request in requests] == [
        'entity-a',
        'entity-b',
    ]
    assert requests[0].model_input.term == 'Alpha'
    assert requests[0].model_input.target_block.content == 'first'
    assert requests[0].model_input.context_before == []
    assert len(requests[0].model_input.context_after) == 1
    assert requests[0].model_input.context_after[0].content == 'second'


def test_entity_enrichment_embedding_preserves_occurrence_identity():
    request = EntityEnrichmentRequest(
        target=EntityEnrichmentTarget(
            uuid='entity-1',
            source_uuid='source-1',
            source_block_uuid='block-1',
            name='Alpha',
        ),
        model_input=TermEnrichmentInput(
            term='Alpha',
            target_block={
                'block_type': 'paragraph',
                'content': 'Alpha appears here',
            },
        ),
    )
    enrichment = EntityEnrichmentNode(_EntityEnricher())
    description_state = asyncio.run(
        enrichment.worker({'entity_enrichment_request': request})
    )
    description_result = description_state['entity_description_results'][0]
    client = _EmbeddingClient()
    embedding = EntityEmbeddingNode(client)

    embedded_state = asyncio.run(
        embedding.worker({'entity_description_results': [description_result]})
    )

    assert client.texts == ['Alpha : description of Alpha']
    result = embedded_state['entity_embedding_results'][0]
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
            }
        ]

    async def consume(self):
        return None


class _RowsSession:
    def __init__(self):
        self.calls = []

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return _RowsResult()


def test_semantic_repository_typed_reads_and_updates_are_source_scoped():
    session = _RowsSession()
    repository = SemanticRepository(lambda: _SessionContext(session))

    entities = asyncio.run(repository.load_entities('source-1'))
    result = EntityEnrichmentResult(
        **entities[0].model_dump(),
        description='a local description',
        embedding=[0.1, 0.2],
    )
    asyncio.run(repository.update_entity_enrichment('source-1', [result]))

    assert session.calls[0] == (
        READ_SOURCE_ENTITIES,
        {'source_uuid': 'source-1'},
    )
    assert session.calls[1][0] is UPDATE_ENTITY_ENRICHMENT
    assert session.calls[1][1] == {
        'source_uuid': 'source-1',
        'rows': [
            {
                'uuid': 'entity-1',
                'description': 'a local description',
                'embedding': [0.1, 0.2],
            }
        ],
    }
    assert 'MATCH (entity:Entity' in UPDATE_ENTITY_ENRICHMENT
    assert 'source_uuid: $source_uuid' in UPDATE_ENTITY_ENRICHMENT
    assert 'HAS_TRIPLET' not in UPDATE_ENTITY_ENRICHMENT


def test_semantic_graph_runs_all_typed_phases_in_one_graph():
    source_repository = _TypedSourceRepository(
        [_block('block-1', 'target text')]
    )
    semantic_repository = _TypedSemanticRepository(
        [
            Entity(
                uuid='entity-1',
                source_uuid='source-1',
                source_block_uuid='block-1',
                name='Alpha',
            )
        ],
        [
            Event(
                uuid='event-1',
                source_uuid='source-1',
                source_block_uuid='block-1',
                name='Appears',
            )
        ],
        [
            Predicate(
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
    )

    graph = SemanticGraph(
        triplet_source_load=TripletSourceLoadNode(source_repository),
        fact_extraction=FactExtractionNode(_FactExtractor(), context_window),
        triplet_decomposition=TripletDecompositionNode(_TripletDecomposer()),
        triplet_persistence=TripletPersistenceNode(
            semantic_repository,
            _noop_schema,
        ),
        entity_enrichment_load=EntityEnrichmentLoadNode(
            source_repository,
            semantic_repository,
            context_window,
        ),
        entity_enrichment=EntityEnrichmentNode(_EntityEnricher()),
        entity_embedding=EntityEmbeddingNode(_EmbeddingClient()),
        entity_enrichment_persistence=EntityEnrichmentPersistenceNode(
            semantic_repository,
            _noop_schema,
        ),
        event_enrichment_load=EventEnrichmentLoadNode(
            source_repository,
            semantic_repository,
            context_window,
        ),
        event_enrichment=EventEnrichmentNode(_EntityEnricher()),
        event_embedding=EventEmbeddingNode(_EmbeddingClient()),
        event_enrichment_persistence=EventEnrichmentPersistenceNode(
            semantic_repository,
            _noop_schema,
        ),
        predicate_enrichment_load=PredicateEnrichmentLoadNode(
            source_repository,
            semantic_repository,
            context_window,
        ),
        predicate_enrichment=PredicateEnrichmentNode(_EntityEnricher()),
        predicate_embedding=PredicateEmbeddingNode(_EmbeddingClient()),
        predicate_enrichment_persistence=PredicateEnrichmentPersistenceNode(
            semantic_repository,
            _noop_schema,
        ),
    ).build_graph()

    assert {
        'triplet_persistence',
        'entity_enrichment_load',
        'event_enrichment_load',
        'predicate_enrichment_load',
    } <= set(graph.nodes)
    final_state = asyncio.run(graph.ainvoke({'source_uuid': 'source-1'}))

    assert len(final_state['raw_assertions']) == 1
    assert final_state['entity_persisted_count'] == 1
    assert final_state['event_persisted_count'] == 1
    assert final_state['predicate_persisted_count'] == 1
    assert set(semantic_repository.updated) == {'entity', 'event', 'predicate'}
