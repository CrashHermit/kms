import asyncio
from types import SimpleNamespace

import dspy
import pytest
from PIL import Image

from kms import config
from kms.construction import instruction_finder
from kms.core import content, models, walker


class _ScriptedModule:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


def _nodes():
    return [
        models.Node(
            type='paragraph',
            content='For the following exercises, simplify.',
            id=10,
        ),
        models.Node(type='image', image_path='a.png', id=20),
        models.Node(type='list', content='3 matrix A', id=30),
        models.Node(
            type='paragraph', content='For the next exercises, solve.', id=40
        ),
        models.Node(type='list', content='4 matrix B', id=50),
    ]


def _view(position, content_text, node_type='paragraph'):
    return walker.WindowNode(
        position=position, type=node_type, content=content_text
    )


def test_specialized_prompts_and_demos_cover_boundary_contract():
    language_model = dspy.LM('openai/dummy', api_key='x')
    router = instruction_finder.InstructionRouter(language_model)
    grower = instruction_finder.InstructionGrower(language_model)

    assert len(router.predictor.demos) == 5
    assert len(grower.predictor.demos) == 7
    assert 'Answer only the boolean True or False.' in (
        instruction_finder.InstructionRouterSignature.__doc__
    )
    assert '<designated>' in instruction_finder.InstructionRouterSignature.__doc__
    assert '<candidate>' in instruction_finder.InstructionGrowerSignature.__doc__
    assert 'Answer only the boolean True or False.' in (
        instruction_finder.InstructionGrowerSignature.__doc__
    )

    router_text = '\\n'.join(
        demo.current_node.content.render() for demo in router.predictor.demos
    )
    assert '185. Determine whether the series converges.' in router_text
    assert 'a. Find the tangent plane.' in router_text
    assert 'For Exercises 8–10' in router_text

    grower_text = '\\n'.join(
        [
            demo.instruction_nodes.content.render()
            + ' | '
            + demo.next_node.content.render()
            for demo in grower.predictor.demos
        ]
    )
    assert '302. $z = 4x^2 + y^2$' in grower_text
    assert 'Decorative publisher illustration.' in grower_text
    assert 'Diagram of the curve and its marked extrema.' in grower_text
    assert [demo.include_next_node for demo in grower.predictor.demos] == [
        True,
        True,
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


def test_router_and_grower_encode_labeled_multimodal_parts(tmp_path):
    image_path = tmp_path / 'figure.png'
    Image.new('RGB', (8, 8), (0, 0, 255)).save(image_path)
    image_node = _view(1, None, 'image')
    image_node.image_path = str(image_path)
    text_node = _view(0, 'For the following exercises, simplify.')

    router_input = instruction_finder.InstructionRouter.encode(
        object(), [image_node]
    )['current_node']
    grower_inputs = instruction_finder.InstructionGrower.encode(
        object(), [text_node], [image_node]
    )

    assert isinstance(router_input, content.ContentParts)
    assert router_input.content.parts[0].text == '[1] (image)'
    assert isinstance(router_input.content.parts[1], content.ImagePart)
    assert isinstance(grower_inputs['instruction_nodes'], content.ContentParts)
    assert isinstance(grower_inputs['next_node'], content.ContentParts)


def test_scan_routes_designated_nodes_and_grows_one_node_at_a_time():
    router = _ScriptedModule([True, False, False, False])
    grower = _ScriptedModule([True, False])

    result = asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes()[:4], router=router, grower=grower
        )
    )

    assert result == [[10, 20]]
    assert [
        [node.position for node in call['current_node']]
        for call in router.calls
    ] == [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert [
        next(
            index
            for index, node in enumerate(call['current_node'])
            if node.marker == 'designated'
        )
        for call in router.calls
    ] == [0, 2, 3]
    assert [
        [node.position for node in call['instruction_nodes']]
        for call in grower.calls
    ] == [[0], [0, 1]]
    assert [
        [node.position for node in call['next_node']]
        for call in grower.calls
    ] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert [
        [node.marker for node in call['next_node']]
        for call in grower.calls
    ] == [
        [None, 'candidate', None, None],
        [None, None, 'candidate', None],
    ]


def test_context_windows_keep_markers_within_the_configured_token_budget():
    nodes = [
        models.Node(
            type='paragraph',
            content='context ' * 30,
            id=index,
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

    router_windows = [call['current_node'] for call in router.calls]
    context_budget = (
        config.get_settings().stages.finders.instruction_finder.context_budget
    )
    grower_windows = [call['next_node'] for call in grower.calls]
    for window in router_windows + grower_windows:
        assert sum(walker.estimate_text_tokens(node.content) for node in window) <= (
            context_budget
        )
        assert sum(node.marker is not None for node in window) == 1
    assert grower_windows[0][1].marker == 'candidate'


def test_scan_finds_multiple_instructions_in_document_order():
    router = _ScriptedModule([True, False, True, False, False])
    grower = _ScriptedModule([False, False])

    result = asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes(), router=router, grower=grower
        )
    )

    assert result == [[10], [30]]


def test_false_growth_banks_only_the_anchor_and_reconsiders_candidate():
    router = _ScriptedModule([True, False, False, False, False])
    grower = _ScriptedModule([False])

    result = asyncio.run(
        instruction_finder.find_instruction_spans(
            _nodes(), router=router, grower=grower
        )
    )

    assert result == [[10]]
    assert [
        node.marker for node in grower.calls[0]['next_node']
    ] == [None, 'candidate', None, None, None]
    assert [
        next(
            node.position
            for node in call['current_node']
            if node.marker == 'designated'
        )
        for call in router.calls
    ] == [0, 1, 2, 3, 4]


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
    out = asyncio.run(node.run({'nodes': nodes, 'source_key': 'book.pdf'}))

    assert set(out) == {'instructions'}
    assert len(out['instructions']) == 1
    instruction = out['instructions'][0]
    assert instruction.block == [10, 20]
    assert instruction.members == [10, 20]
    assert [item.id for item in nodes] == [10, 20, 30, 40, 50]
