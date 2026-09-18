import asyncio

from langgraph.graph import START, StateGraph

from kms2.config.semantic import SourceEntityHubSettings
from kms2.core.model import (
    SourceEntityHubCandidate,
    SourceEntityHubDefinition,
    SourceEntityHubRerankResult,
    SourceEntityHubSynthesisResult,
)
from kms2.langgraph.semantic.source_entity_hub import (
    add_source_entity_hub_phase,
)
from kms2.langgraph.semantic.source_event_hub import add_source_event_hub_phase
from kms2.langgraph.semantic.source_predicate_hub import (
    add_source_predicate_hub_phase,
)
from kms2.langgraph.semantic.source_procedure_hub import (
    add_source_procedure_hub_phase,
)
from kms2.langgraph.semantic.source_statement_hub import (
    add_source_statement_hub_phase,
)
from kms2.langgraph.semantic.source_triplet_hub import (
    add_source_triplet_hub_phase,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_entity_hub import SourceEntityHubNode


class _TraceNode:
    def __init__(self, kind, trace):
        self.kind = kind
        self.trace = trace

    def _mark(self, phase):
        self.trace.append(f'{self.kind}:{phase}')

    async def load_candidates(self, state):
        self._mark('load')
        return {f'{self.kind}_candidates': []}

    def dispatch_rerank(self, state):
        return f'{self.kind}_rerank_collect'

    async def rerank_worker(self, state):
        return {}

    def collect_rerank(self, state):
        self._mark('rerank_collect')
        return {}

    def dispatch_judge(self, state):
        return f'{self.kind}_judge_collect'

    async def judge_worker(self, state):
        return {}

    def collect_judge(self, state):
        self._mark('judge_collect')
        return {}

    async def detect_communities(self, state):
        self._mark('communities')
        return {f'{self.kind}_communities': []}

    def dispatch_synthesis(self, state):
        return f'{self.kind}_synthesis_collect'

    async def synthesis_worker(self, state):
        return {}

    def collect_synthesis(self, state):
        self._mark('synthesis_collect')
        return {}

    async def embed(self, state):
        self._mark('embedding')
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
        self._mark('load')
        return {'source_triplet_hub_groups': []}

    async def run(self, state):
        self._mark('persist')
        return {}


class _Persistence:
    def __init__(self, marker, trace):
        self.marker = marker
        self.trace = trace

    async def run(self, state):
        self.trace.append(self.marker)
        return {}


def _phase_graph(trace):
    graph = StateGraph(SemanticState)
    kinds = ('entity', 'event', 'predicate', 'statement', 'procedure')
    for kind in kinds:
        name = f'source_{kind}_persistence'
        graph.add_node(
            name, _Persistence(f'{kind}:description_persist', trace).run
        )
    graph.add_edge(START, 'source_entity_persistence')
    for previous, current in zip(kinds, kinds[1:], strict=False):
        graph.add_edge(
            f'source_{previous}_persistence',
            f'source_{current}_persistence',
        )

    hubs = {
        kind: _TraceNode(f'source_{kind}_hub', trace)
        for kind in ('entity', 'event', 'predicate', 'statement', 'procedure')
    }
    triplet = _TraceNode('source_triplet_hub', trace)
    persistence = {
        kind: _Persistence(f'{kind}:hub_persist', trace)
        for kind in (
            'entity',
            'event',
            'predicate',
            'statement',
            'procedure',
            'triplet',
        )
    }
    add_source_entity_hub_phase(graph, hubs['entity'], persistence['entity'])
    add_source_event_hub_phase(graph, hubs['event'], persistence['event'])
    add_source_predicate_hub_phase(
        graph, hubs['predicate'], persistence['predicate']
    )
    add_source_statement_hub_phase(
        graph, hubs['statement'], persistence['statement']
    )
    add_source_procedure_hub_phase(
        graph, hubs['procedure'], persistence['procedure']
    )
    add_source_triplet_hub_phase(graph, triplet, persistence['triplet'])
    graph.add_edge(
        'source_procedure_persistence',
        'source_entity_hub_load',
    )
    graph.add_edge('source_entity_hub_persistence', 'source_event_hub_load')
    graph.add_edge('source_event_hub_persistence', 'source_predicate_hub_load')
    graph.add_edge(
        'source_predicate_hub_persistence', 'source_statement_hub_load'
    )
    graph.add_edge(
        'source_statement_hub_persistence', 'source_procedure_hub_load'
    )
    graph.add_edge(
        'source_procedure_hub_persistence', 'source_triplet_hub_load'
    )
    return graph.compile()


class _Embedding:
    async def embed(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


def test_entity_collectors_restore_ordinal_order_after_reversed_workers():
    first = SourceEntityHubCandidate(
        left_uuid='left-1',
        left_name='Alpha',
        left_description='first',
        right_uuid='right-1',
        right_name='Beta',
        right_description='first',
        score=0.9,
    )
    second = first.model_copy(
        update={
            'left_uuid': 'left-2',
            'right_uuid': 'right-2',
            'left_name': 'Beta',
            'right_name': 'B',
        }
    )
    node = SourceEntityHubNode(
        None,
        None,
        None,
        None,
        _Embedding(),
        SourceEntityHubSettings(),
    )
    state = SemanticState(
        source_uuid='source-1',
        source_entity_hub_rerank_results=[
            SourceEntityHubRerankResult(ordinal=1, direct=[second]),
            SourceEntityHubRerankResult(ordinal=0, direct=[first]),
        ],
    )
    collected = node.collect_rerank(state)
    assert [
        item.right_uuid for item in collected['source_entity_hub_direct_pairs']
    ] == [
        'right-1',
        'right-2',
    ]
    state = state.model_copy(
        update={
            'source_entity_hub_synthesis_results': [
                SourceEntityHubSynthesisResult(
                    ordinal=1,
                    definition=SourceEntityHubDefinition(
                        canonical_name='Beta', description='second'
                    ),
                    membership_uuids=['right-2'],
                    aliases=['B'],
                ),
                SourceEntityHubSynthesisResult(
                    ordinal=0,
                    definition=SourceEntityHubDefinition(
                        canonical_name='Alpha', description='first'
                    ),
                    membership_uuids=['right-1'],
                    aliases=['A'],
                ),
            ]
        }
    )
    state = state.model_copy(update=node.collect_synthesis(state))
    result = asyncio.run(node.embed(state))
    assert [hub.canonical_name for hub in result['source_entity_hubs']] == [
        'Alpha',
        'Beta',
    ]


def test_typed_hub_stages_are_serial_after_description_persistence():
    trace = []
    graph = _phase_graph(trace)

    asyncio.run(graph.ainvoke({'source_uuid': 'source-1'}))
    for previous, current in (
        ('entity', 'event'),
        ('event', 'predicate'),
        ('predicate', 'statement'),
        ('statement', 'procedure'),
    ):
        assert trace.index(f'{previous}:description_persist') < trace.index(
            f'{current}:description_persist'
        )

    assert max(
        trace.index(f'{kind}:description_persist')
        for kind in ('entity', 'event', 'predicate', 'statement', 'procedure')
    ) < trace.index('source_entity_hub:load')
    stages = (
        'load',
        'rerank_collect',
        'judge_collect',
        'communities',
        'synthesis_collect',
        'embedding',
    )
    for kind in ('entity', 'event', 'predicate', 'statement', 'procedure'):
        positions = [
            trace.index(f'source_{kind}_hub:{stage}') for stage in stages
        ]
        assert positions == sorted(positions)
    for previous, current in (
        ('entity', 'event'),
        ('event', 'predicate'),
        ('predicate', 'statement'),
        ('statement', 'procedure'),
    ):
        assert trace.index(f'{previous}:hub_persist') < trace.index(
            f'source_{current}_hub:load'
        )
    assert trace.index('procedure:hub_persist') < trace.index(
        'source_triplet_hub:load'
    )
    assert trace.index('source_triplet_hub:embedding') < trace.index(
        'triplet:hub_persist'
    )
