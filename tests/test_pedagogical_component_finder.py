import asyncio

import pytest

from kms.construction import pedagogical_component_finder
from kms.core import context_window, models


class _ScriptedRouter:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    async def aforward(self, **inputs):
        self.calls.append(inputs)
        return self._scripted.pop(0)


def _nodes(count=4, tokens=1):
    return [
        models.SourceNode(
            type='paragraph',
            content=f'node-{index} ' + 'x' * (4 * (tokens - 1)),
            uuid=f'node-{index}',
        )
        for index in range(count)
    ]


def _budgets(value=300):
    return {
        'start_before_budget': value,
        'start_after_budget': value,
        'end_before_budget': value,
        'end_after_budget': value,
    }


def _find(nodes, start_values, end_values, **budgets):
    start_router = _ScriptedRouter(start_values)
    end_router = _ScriptedRouter(end_values)
    spans = asyncio.run(
        pedagogical_component_finder.find_spans(
            nodes,
            start_router=start_router,
            end_router=end_router,
            **(_budgets() | budgets),
        )
    )
    return spans, start_router, end_router


def test_false_starts_advance_one_node_without_end_calls():
    spans, start_router, end_router = _find(
        _nodes(3), [False, False, True], [True]
    )

    assert spans == [[2]]
    assert [
        call['target_node'].text.strip() for call in start_router.calls
    ] == [
        'node-0',
        'node-1',
        'node-2',
    ]
    assert len(end_router.calls) == 1


def test_end_router_advances_candidate_and_resumes_after_committed_span():
    spans, start_router, end_router = _find(
        _nodes(4), [True, False], [False, False, True]
    )

    assert spans == [[0, 1, 2]]
    assert [
        call['candidate_node'].text.strip() for call in end_router.calls
    ] == [
        'node-0',
        'node-1',
        'node-2',
    ]
    assert [
        call['target_node'].text.strip() for call in start_router.calls
    ] == [
        'node-0',
        'node-3',
    ]


def test_two_units_are_ordered_and_interior_nodes_skip_start_router():
    spans, start_router, end_router = _find(
        _nodes(4), [True, False, True], [False, True, True]
    )

    assert spans == [[0, 1], [3]]
    assert [
        call['target_node'].text.strip() for call in start_router.calls
    ] == [
        'node-0',
        'node-2',
        'node-3',
    ]


def test_eof_candidate_has_empty_after_context():
    spans, _, end_router = _find(_nodes(1), [True], [True])

    assert spans == [[0]]
    assert end_router.calls[0]['context_after'] == []
    assert end_router.calls[0]['candidate_node'].index == 1


def test_false_eof_end_raises_exact_boundary_error():
    with pytest.raises(
        ValueError,
        match=r'^pedagogical unit starting at cursor 0 has no end before the node stream ends$',
    ):
        _find(_nodes(1), [True], [False])


@pytest.mark.parametrize('value', ['true', 1, None])
def test_start_decoder_rejects_non_boolean_predictions(value):
    router = object.__new__(pedagogical_component_finder.PedagogicalStartRouter)
    prediction = type('Prediction', (), {'is_pedagogical_start': value})()

    with pytest.raises(
        ValueError, match='is_pedagogical_start must be a boolean'
    ):
        router.decode(prediction)


@pytest.mark.parametrize('value', ['false', 0, None])
def test_end_decoder_rejects_non_boolean_predictions(value):
    router = object.__new__(pedagogical_component_finder.PedagogicalEndRouter)
    prediction = type('Prediction', (), {'is_pedagogical_end': value})()

    with pytest.raises(
        ValueError, match='is_pedagogical_end must be a boolean'
    ):
        router.decode(prediction)


def test_router_contexts_respect_directional_token_budgets_and_local_indexes():
    spans, start_router, end_router = _find(
        _nodes(5, tokens=2),
        [False, False, True, False, False],
        [True],
        start_before_budget=2,
        start_after_budget=2,
        end_before_budget=2,
        end_after_budget=2,
    )

    assert spans == [[2]]
    start_call = start_router.calls[2]
    end_call = end_router.calls[0]
    for call in (start_call, end_call):
        assert (
            sum(
                context_window.estimate_text_tokens(item.text)
                for item in call['context_before']
            )
            <= 2
        )
        assert (
            sum(
                context_window.estimate_text_tokens(item.text)
                for item in call['context_after']
            )
            <= 2
        )
        assert [item.index for item in call['context_before']] == list(
            range(1, len(call['context_before']) + 1)
        )
        assert [item.index for item in call['context_after']] == list(
            range(1, len(call['context_after']) + 1)
        )
        assert (
            call['target_node']
            if 'target_node' in call
            else call['candidate_node']
        ).index == 1
    assert end_call['start_node'].index == 1
    assert all(
        not hasattr(item, field)
        for item in [start_call['target_node'], end_call['start_node']]
        for field in ('assets', 'path')
    )


def test_instruction_members_are_excluded_before_span_remapping():
    start_router = _ScriptedRouter([True, False])
    end_router = _ScriptedRouter([False, True])
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        start_router=start_router,
        end_router=end_router,
    )

    out = asyncio.run(
        node.run(
            {
                'nodes': _nodes(),
                'instructions': [
                    models.Instruction(block=[0], member_positions=[0])
                ],
            }
        )
    )

    assert out == {'spans': [[1, 2]]}


def test_empty_input_yields_empty_spans_channel():
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        start_router=_ScriptedRouter([]),
        end_router=_ScriptedRouter([]),
    )

    assert asyncio.run(node.run({'nodes': []})) == {'spans': []}


def test_router_encoders_keep_text_only_typed_fields():
    before = [models.NodeInput(index=1, node_type='paragraph', text='before')]
    target = models.NodeInput(index=1, node_type='paragraph', text='target')
    after = [models.NodeInput(index=1, node_type='paragraph', text='after')]
    start_router = object.__new__(
        pedagogical_component_finder.PedagogicalStartRouter
    )
    end_router = object.__new__(
        pedagogical_component_finder.PedagogicalEndRouter
    )

    assert start_router.encode(before, target, after) == {
        'context_before': before,
        'target_node': target,
        'context_after': after,
    }
    assert end_router.encode(target, before, target, after) == {
        'start_node': target,
        'context_before': before,
        'candidate_node': target,
        'context_after': after,
    }
