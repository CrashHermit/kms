import asyncio
from types import SimpleNamespace

import pytest
from langgraph.graph import END, StateGraph

from kms2.config.inference import ContextWindowSettings
from kms2.config.settings import Settings
from kms2.core.model.block import SourceBlock
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_extraction import SourceExtractionTarget
from kms2.core.model.source_semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionInput,
    SourceEntityDescriptionOutput,
    SourceEntityDescriptionRequest,
    SourceEntityDescriptionResult,
    SourceEntityDescriptionTarget,
)
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.source_semantic.source_entity_description import (
    add_source_entity_description_phase,
)
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

_GLOSSES = {
    'alpha': 'First source-local description.',
    'beta': 'A distinct source-local description.',
    'gamma': 'Third independently grounded description.',
}


def _blocks():
    return [
        SourceBlock(uuid='block-1', block_type='paragraph', content='one'),
        SourceBlock(uuid='block-2', block_type='paragraph', content='two'),
        SourceBlock(uuid='block-3', block_type='paragraph', content='three'),
    ]


class _TokenCounter:
    def count_texts(self, texts):
        return [int(bool(text)) for text in texts]


_TOKEN_COUNTERS = (_TokenCounter(),)


def test_entity_description_uses_one_owner_and_transient_context():
    blocks = _blocks()
    occurrence = SourceEntity(
        uuid='entity-1',
        source_uuid='source-1',
        name='Matrix A',
        target=SourceExtractionTarget(source_blocks=[blocks[0]]),
    )

    class EntityRepository:
        async def load_source_entities(self, source_uuid):
            return [occurrence]

    node = SourceEntityDescriptionLoadNode(
        EntityRepository(),
        ContextWindowSettings(backward_budget=1, forward_budget=1),
        token_counters=_TOKEN_COUNTERS,
    )
    result = asyncio.run(
        node.run(SourceSemanticState(source_uuid='source-1', blocks=blocks))
    )
    request = result['source_entity_description_requests'][0]

    assert request.model_input.term == 'Matrix A'
    assert [block.content for block in request.model_input.target_blocks] == [
        'one',
    ]
    assert [block.content for block in request.model_input.context_after] == [
        'two',
    ]
    assert request.target.target.source_blocks[0].uuid == 'block-1'


def _description_case():
    requests = []
    for index, content in enumerate(_GLOSSES):
        owner = SourceBlock(
            uuid=f'owner-{index}', block_type='paragraph', content=content
        )
        requests.append(
            SourceEntityDescriptionRequest(
                target=SourceEntityDescriptionTarget(
                    uuid=f'entity-{index}',
                    source_uuid='source-1',
                    name='Same name',
                    target=SourceExtractionTarget(source_blocks=[owner]),
                ),
                model_input=SourceEntityDescriptionInput(
                    term='Same name',
                    context_before=[
                        SourceBlockContext(
                            block_type='paragraph', content=f'before {content}'
                        )
                    ],
                    target_blocks=[
                        SourceBlockContext(
                            block_type=owner.block_type, content=owner.content
                        )
                    ],
                    context_after=[
                        SourceBlockContext(
                            block_type='paragraph', content=f'after {content}'
                        )
                    ],
                ),
            )
        )
    return SimpleNamespace(
        requests=requests,
        request_key='source_entity_description_requests',
        result_key='source_entity_description_results',
        collector='source_entity_description_collect',
    )


def _description_state(**fields):
    return SourceSemanticState(source_uuid='source-1', **fields)


class _BatchDescriber:
    def __init__(self):
        self.calls = []

    async def acall(self, *, items):
        self.calls.append(items)
        return [
            SourceEntityDescriptionOutput(
                position=item.position,
                description=_GLOSSES[item.request.target_blocks[0].content],
            )
            for item in reversed(items)
        ]


def test_entity_description_batches_restore_owners_and_independent_evidence():
    case = _description_case()
    originals = [request.model_copy(deep=True) for request in case.requests]
    outputs = []
    for size in (16, 1):
        describer = _BatchDescriber()
        node = SourceEntityDescriptionNode(
            describer,
            budget=TokenBudget(_TokenCounter(), 100),
            batch_size=size,
        )
        sends = node.dispatch(
            _description_state(**{case.request_key: case.requests})
        )

        async def run_packets(node=node, sends=sends):
            results = []
            for send in sends:
                results.extend((await node.worker(send.arg))[case.result_key])
            return results

        results = asyncio.run(run_packets())
        outputs.append(results)
        assert len(describer.calls) == (1 if size == 16 else 3)
        for call in describer.calls:
            assert [item.position for item in call] == list(range(len(call)))
        assert [result.description for result in results] == list(
            _GLOSSES.values()
        )
        assert [
            result.model_dump(exclude={'description', 'embedding'})
            for result in results
        ] == [request.target.model_dump() for request in case.requests]
        projected = [item.request for call in describer.calls for item in call]
        assert projected == [request.model_input for request in case.requests]
    assert outputs[0] == outputs[1]
    assert case.requests == originals


@pytest.mark.parametrize(
    ('token_limit', 'batch_size', 'expected_sizes'),
    [(7, 16, [3]), (6, 16, [2, 1]), (7, 1, [1, 1, 1])],
)
def test_entity_description_packets_respect_text_and_count_limits(
    token_limit, batch_size, expected_sizes
):
    case = _description_case()
    node = SourceEntityDescriptionNode(
        _BatchDescriber(),
        budget=TokenBudget(_TokenCounter(), token_limit),
        batch_size=batch_size,
    )
    sends = node.dispatch(
        _description_state(**{case.request_key: case.requests})
    )
    assert [len(send.arg[case.request_key]) for send in sends] == expected_sizes
    assert [
        request for send in sends for request in send.arg[case.request_key]
    ] == case.requests


def test_oversized_entity_description_is_not_split_or_sent_for_inference():
    case = _description_case()
    describer = _BatchDescriber()
    node = SourceEntityDescriptionNode(
        describer, budget=TokenBudget(_TokenCounter(), 2), batch_size=16
    )
    with pytest.raises(InputBudgetExceeded, match='singleton'):
        node.dispatch(_description_state(**{case.request_key: case.requests}))
    assert describer.calls == []


def test_no_entity_description_requests_skip_counting_and_inference():
    class UnusedCounter:
        def count_texts(self, texts):
            raise AssertionError('empty work must not count tokens')

    describer = _BatchDescriber()
    node = SourceEntityDescriptionNode(
        describer, budget=TokenBudget(UnusedCounter(), 100), batch_size=16
    )
    assert (
        node.dispatch(_description_state())
        == 'source_entity_description_collect'
    )
    assert describer.calls == []


def test_entity_description_graph_fan_in_preserves_request_order():
    case = _description_case()
    completed = []

    class Loader:
        async def run(self, state):
            return {case.request_key: case.requests}

    async def run_graph():
        released = asyncio.Event()

        class DelayedDescriber(_BatchDescriber):
            async def acall(self, *, items):
                first = items[0].request.target_blocks[0].content
                if first == 'alpha':
                    await released.wait()
                else:
                    released.set()
                completed.append(first)
                return await super().acall(items=items)

        node = SourceEntityDescriptionNode(
            DelayedDescriber(),
            budget=TokenBudget(_TokenCounter(), 100),
            batch_size=2,
        )
        graph = StateGraph(SourceSemanticState)
        graph.add_node('source_entity_extraction_persistence', lambda state: {})
        graph.set_entry_point('source_entity_extraction_persistence')
        add_source_entity_description_phase(graph, Loader(), node)
        graph.add_edge(case.collector, END)
        result = await graph.compile().ainvoke(_description_state())
        return result[case.result_key]

    results = asyncio.run(run_graph())
    assert completed == ['gamma', 'alpha']
    assert [result.uuid for result in results] == [
        request.target.uuid for request in case.requests
    ]
    assert [result.description for result in results] == list(_GLOSSES.values())


def test_entity_embedding_preserves_occurrence_identity_and_order():
    described = [
        SourceEntityDescriptionResult(
            **request.target.model_dump(),
            description=_GLOSSES[request.model_input.target_blocks[0].content],
        )
        for request in _description_case().requests
    ]

    class EmbeddingClient:
        async def embed(self, texts):
            return [[float(index)] for index, _ in enumerate(texts)]

    embedded = asyncio.run(
        SourceEntityEmbeddingNode(EmbeddingClient()).run(
            _description_state(source_entity_description_results=described)
        )
    )['source_entity_embedding_results']
    assert [(result.uuid, result.embedding) for result in embedded] == [
        (f'entity-{index}', [float(index)]) for index in range(3)
    ]


@pytest.mark.parametrize(
    ('count_limit', 'token_limit', 'expected'),
    [(1, 7, [1, 1, 1]), (16, 6, [2, 1])],
)
def test_entity_description_settings_control_whole_request_packets(
    count_limit, token_limit, expected, monkeypatch
):
    case = _description_case()
    prefix = 'KMS2_SOURCE_SEMANTIC__SOURCE_ENTITY_DESCRIPTION__'
    monkeypatch.setenv(prefix + 'BATCH_SIZE', str(count_limit))
    monkeypatch.setenv(prefix + 'INPUT_TOKEN_BUDGET', str(token_limit))
    monkeypatch.setenv(prefix + 'SAFETY_MARGIN_TOKENS', '0')
    monkeypatch.setenv(prefix + 'INFERENCE__PROFILE', 'chatgpt-luna')
    settings = Settings(_env_file=None)
    config = settings.source_semantic.source_entity_description
    node = SourceEntityDescriptionNode(
        _BatchDescriber(),
        budget=TokenBudget(_TokenCounter(), config.input_token_budget),
        batch_size=config.batch_size,
    )
    sends = node.dispatch(
        _description_state(**{case.request_key: case.requests})
    )
    assert [len(send.arg[case.request_key]) for send in sends] == expected
