import asyncio
from types import SimpleNamespace

import dspy
import pytest

from kms import config
from kms.construction import instruction_finder
from kms.core import context_window, models


class _ScriptedModule:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


def _nodes():
    return [
        models.SourceNode(
            type='paragraph',
            content='For the following exercises, simplify.',
            uuid='node-10',
        ),
        models.SourceNode(
            type='image',
            assets=[models.VisualAsset(path='a.png')],
            uuid='node-20',
        ),
        models.SourceNode(type='list', content='3 matrix A', uuid='node-30'),
        models.SourceNode(
            type='paragraph',
            content='For the next exercises, solve.',
            uuid='node-40',
        ),
        models.SourceNode(type='list', content='4 matrix B', uuid='node-50'),
    ]


def _view(position, content_text, node_type='paragraph'):
    return context_window.ContextNode(
        position=position, type=node_type, content=content_text
    )


def test_specialized_prompts_and_demos_cover_boundary_contract():
    language_model = dspy.LM('openai/dummy', api_key='x')
    router = instruction_finder.InstructionRouter(language_model)
    grower = instruction_finder.InstructionGrower(language_model)

    assert len(router.predictor.demos) == 5
    assert len(grower.predictor.demos) == 9
    assert 'Answer only the boolean True or False.' in (
        instruction_finder.InstructionRouterSignature.__doc__
    )
    assert (
        'target_node' in instruction_finder.InstructionRouterSignature.__doc__
    )
    assert (
        'candidate_node'
        in instruction_finder.InstructionGrowerSignature.__doc__
    )
    router_text = '\n'.join(
        demo.target_node.text for demo in router.predictor.demos
    )
    assert '185. Determine whether the series converges.' in router_text
    assert 'a. Find the tangent plane.' in router_text
    assert 'For Exercises 8–10' in router_text
    grower_text = '\n'.join(
        item.text
        for demo in grower.predictor.demos
        for item in (
            demo.accepted_nodes
            + [demo.candidate_node]
            + demo.context_before
            + demo.context_after
        )
    )
    assert '302. $z = 4x^2 + y^2$' in grower_text
    assert 'Decorative publisher illustration.' in grower_text
    assert 'Find the gradient of f(x, y).' in grower_text
    assert 'Diagram of the curve and its marked extrema.' in grower_text
    assert [demo.include_next_node for demo in grower.predictor.demos] == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_router_and_grower_return_only_strict_booleans():
    router = object.__new__(instruction_finder.InstructionRouter)
    grower = object.__new__(instruction_finder.InstructionGrower)

    assert (
        instruction_finder.InstructionRouter.decode(
            router, SimpleNamespace(is_instruction_start=True)
        )
        is True
    )
    assert (
        instruction_finder.InstructionGrower.decode(
            grower, SimpleNamespace(include_next_node=False)
        )
        is False
    )

    with pytest.raises(
        ValueError, match='is_instruction_start must be a boolean'
    ):
        instruction_finder.InstructionRouter.decode(
            router, SimpleNamespace(is_instruction_start='True')
        )
    with pytest.raises(ValueError, match='include_next_node must be a boolean'):
        instruction_finder.InstructionGrower.decode(
            grower, SimpleNamespace(include_next_node=1)
        )

    target = models.NodeInput(
        index=1,
        node_type='image',
        text='A diagram of the curve.',
    )
    before = [models.NodeInput(index=1, node_type='paragraph', text='Before')]
    after = [models.NodeInput(index=1, node_type='paragraph', text='After')]

    router_inputs = instruction_finder.InstructionRouter.encode(
        object(), before, target, after
    )
    grower_inputs = instruction_finder.InstructionGrower.encode(
        object(), [], before, target, after
    )

    assert router_inputs['target_node'].text == ('A diagram of the curve.')
    assert router_inputs['context_before'] == before
    assert router_inputs['context_after'] == after
    assert grower_inputs['candidate_node'] == target
    assert grower_inputs['accepted_nodes'] == []
    assert not hasattr(target, 'assets')


def test_scan_routes_designated_nodes_and_grows_one_node_at_a_time():
    router = _ScriptedModule([True, False, False, False])
    grower = _ScriptedModule([True, False])

    result = asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes()[:4], router=router, grower=grower
        )
    )

    assert result == [[0, 1]]
    assert [
        [node.index for node in call['context_after']] for call in router.calls
    ] == [[1, 2, 3], [1], []]
    assert [call['target_node'].index for call in router.calls] == [1, 1, 1]
    assert all(call['context_before'] == [] for call in router.calls)
    assert [
        [node.index for node in call['accepted_nodes']] for call in grower.calls
    ] == [[1], [1, 2]]
    assert [call['candidate_node'].index for call in grower.calls] == [1, 1]
    assert [
        [node.index for node in call['context_after']] for call in grower.calls
    ] == [[1, 2], [1]]


def test_context_windows_keep_context_within_the_configured_token_budget():
    nodes = [
        models.SourceNode(
            type='paragraph',
            content='context ' * 30,
            uuid=f'node-{index}',
        )
        for index in range(20)
    ]
    router = _ScriptedModule([True] + [False] * 19)
    grower = _ScriptedModule([False])

    asyncio.run(
        instruction_finder.find_instruction_spans(
            nodes, router=router, grower=grower
        )
    )

    context_budget = (
        config.get_settings().stages.finders.instruction_finder.context_budget
    )
    context_lists = [call['context_after'] for call in router.calls] + [
        call['context_after'] for call in grower.calls
    ]
    for context_after in context_lists:
        assert (
            sum(
                context_window.estimate_text_tokens(node.text)
                for node in context_after
            )
            <= context_budget
        )
    assert router.calls[0]['target_node'].text.startswith('context')


def test_scan_finds_multiple_instructions_in_document_order():
    router = _ScriptedModule([True, False, True, False, False])
    grower = _ScriptedModule([False, False])

    result = asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes(), router=router, grower=grower
        )
    )

    assert result == [[0], [2]]


def test_false_growth_banks_only_the_anchor_and_reconsiders_candidate():
    router = _ScriptedModule([True, False, False, False, False])
    grower = _ScriptedModule([False])

    asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes(), router=router, grower=grower
        )
    )

    assert grower.calls[0]['candidate_node'].node_type == 'image'
    assert grower.calls[0]['candidate_node'].text == ''
    assert [call['target_node'].index for call in router.calls] == [
        1,
        1,
        1,
        1,
        1,
    ]


def test_instruction_growth_fails_at_instruction_budget(monkeypatch):
    finder_settings = SimpleNamespace(
        instruction_finder=SimpleNamespace(context_budget=10, max_span_budget=3)
    )
    monkeypatch.setattr(
        config,
        'get_settings',
        lambda: SimpleNamespace(
            stages=SimpleNamespace(finders=finder_settings)
        ),
    )
    nodes = [
        models.SourceNode(type='paragraph', content='x' * 8),
        models.SourceNode(type='paragraph', content='y' * 8),
    ]

    with pytest.raises(ValueError, match='look-ahead limit'):
        asyncio.run(
            instruction_finder.find_instruction_spans(
                nodes,
                router=_ScriptedModule([True]),
                grower=_ScriptedModule([]),
            )
        )


def test_malformed_router_boolean_fails_fast():
    with pytest.raises(
        ValueError, match='is_instruction_start must be a boolean'
    ):
        asyncio.run(
            instruction_finder.find_instruction_spans(
                _nodes(),
                router=_ScriptedModule(['yes']),
                grower=_ScriptedModule([]),
            )
        )


def test_malformed_grower_boolean_fails_fast():
    with pytest.raises(ValueError, match='include_next_node must be a boolean'):
        asyncio.run(
            instruction_finder.find_instruction_spans(
                _nodes(),
                router=_ScriptedModule([True]),
                grower=_ScriptedModule([None]),
            )
        )


def test_finder_node_emits_instruction_hubs_without_mutating_nodes():
    node = instruction_finder.InstructionFinderNode(
        router=_ScriptedModule([True, False, False, False, False]),
        grower=_ScriptedModule([True, False]),
    )
    nodes = _nodes()
    out = asyncio.run(
        node.run({'nodes': nodes, 'source': models.Source(key='book.pdf')})
    )

    assert set(out) == {'instructions'}
    assert len(out['instructions']) == 1
    instruction = out['instructions'][0]
    assert instruction.block == [0, 1]
    assert instruction.member_positions == [0, 1]
    assert [item.uuid for item in nodes] == [
        'node-10',
        'node-20',
        'node-30',
        'node-40',
        'node-50',
    ]
