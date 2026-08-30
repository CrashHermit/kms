import asyncio

import pytest

from kms.construction import pedagogical_component_finder
from kms.core import context_window, models, walker


class _ScriptedFinder:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


class _Ready:
    async def aforward(self, current_nodes):
        return True


class _AlwaysUnready:
    async def aforward(self, current_nodes):
        return False


class _TrackingReady:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def aforward(self, current_nodes):
        self.calls.append(len(current_nodes))
        return self.values.pop(0)


def _nodes():
    return [
        models.SourceNode(
            type='paragraph', content='intro prose', uuid='node-0'
        ),
        models.SourceNode(type='header', content='Example 1', uuid='node-1'),
        models.SourceNode(
            type='paragraph', content='solve this', uuid='node-2'
        ),
        models.SourceNode(
            type='paragraph', content='more prose', uuid='node-3'
        ),
    ]


def test_unready_window_grows_before_finder_call():
    readiness = _TrackingReady([False, True, True, True, True])
    finder = _ScriptedFinder([[], [], []])
    assert (
        asyncio.run(
            pedagogical_component_finder.find_spans(
                _nodes(),
                module=finder,
                readiness_module=readiness,
                budget=1,
                max_budget=4,
            )
        )
        == []
    )
    assert readiness.calls[0] == 1
    assert len(finder._scripted) < 3


def test_banks_a_bounded_span_and_emits_member_ids():
    module = _ScriptedFinder([[walker.Span(start=1, end=2)], []])
    spans = asyncio.run(
        pedagogical_component_finder.find_spans(
            _nodes(), module=module, readiness_module=_Ready()
        )
    )
    assert spans == [[1, 2]]


def test_banks_multiple_bounded_spans_in_document_order():
    module = _ScriptedFinder(
        [
            [
                walker.Span(start=1, end=1),
                walker.Span(start=2, end=2),
            ],
            [],
        ]
    )
    assert asyncio.run(
        pedagogical_component_finder.find_spans(
            _nodes(), module=module, readiness_module=_Ready()
        )
    ) == [[1], [2]]


def test_on_prose_only_stream_returns_nothing():
    module = _ScriptedFinder([[]])
    assert (
        asyncio.run(
            pedagogical_component_finder.find_spans(
                _nodes(), module=module, readiness_module=_Ready()
            )
        )
        == []
    )


def test_invalid_span_positions_fail_instead_of_being_clamped():
    for span in (
        walker.Span(start=-1, end=0),
        walker.Span(start=0, end=4),
        walker.Span(start=2, end=1),
    ):
        with pytest.raises(ValueError, match='invalid span'):
            walker.validate_spans([span], 4)


def test_overlapping_spans_fail_instead_of_being_repaired():
    with pytest.raises(ValueError, match='overlapping'):
        walker.validate_spans(
            [walker.Span(start=0, end=2), walker.Span(start=1, end=3)], 4
        )


def test_reversed_span_order_fails_instead_of_being_sorted():
    with pytest.raises(ValueError, match='out-of-order'):
        walker.validate_spans(
            [walker.Span(start=2, end=2), walker.Span(start=0, end=0)], 4
        )


def test_missing_node_id_fails_instead_of_being_dropped():
    # With position-based references, UUIDs are not required during
    # span finding. This test is kept for documentation of the old behavior.
    pass


def test_decoder_deduplicates_exact_spans():
    prediction = type(
        'Prediction',
        (),
        {
            'spans': [
                walker.Span(start=1, end=1),
                walker.Span(start=2, end=2),
                walker.Span(start=2, end=2),
                walker.Span(start=3, end=3),
            ]
        },
    )()
    finder = object.__new__(
        pedagogical_component_finder.PedagogicalComponentFinder
    )
    assert finder.decode(prediction, current_nodes=_nodes()) == [
        walker.Span(start=0, end=0),
        walker.Span(start=1, end=1),
        walker.Span(start=2, end=2),
    ]


def test_decoder_keeps_rejecting_non_identical_overlap():
    prediction = type(
        'Prediction',
        (),
        {
            'spans': [
                walker.Span(start=1, end=2),
                walker.Span(start=2, end=3),
            ]
        },
    )()
    finder = object.__new__(
        pedagogical_component_finder.PedagogicalComponentFinder
    )
    with pytest.raises(ValueError, match='overlapping'):
        finder.decode(prediction, current_nodes=_nodes())


def test_edge_span_at_lookahead_limit_fails_instead_of_being_banked():
    module = _ScriptedFinder([])
    readiness = _AlwaysUnready()
    with pytest.raises(ValueError, match='look-ahead limit'):
        asyncio.run(
            walker.find_spans(
                _nodes(), module, readiness, budget=1, max_budget=1
            )
        )


def test_node_run_writes_the_spans_channel():
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        module=_ScriptedFinder(
            [
                [
                    walker.Span(start=1, end=1),
                    walker.Span(start=2, end=2),
                ],
                [],
            ]
        ),
        readiness_module=_Ready(),
    )
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'spans'}
    assert out['spans'] == [[1], [2]]


def test_node_run_maps_spans_back_after_excluding_instruction_members():
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        module=_ScriptedFinder([[walker.Span(start=0, end=1)], []]),
        readiness_module=_Ready(),
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
    assert out['spans'] == [[1, 2]]


def test_node_run_on_empty_stream_yields_an_empty_channel():
    node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        module=_ScriptedFinder([]),
        readiness_module=_Ready(),
    )
    assert asyncio.run(node.run({'nodes': []})) == {'spans': []}


def test_projection_is_text_only_and_preserves_image_positions():
    nodes = [
        context_window.ContextNode(
            position=0, type='paragraph', content='intro'
        ),
        context_window.ContextNode(
            position=1,
            type='image',
            content='Diagram of the curve.',
            assets=[models.VisualAsset(path='figure.png')],
        ),
    ]

    encoded = pedagogical_component_finder.PedagogicalComponentFinder.encode(
        object(), current_nodes=nodes
    )['current_nodes']

    assert all(isinstance(item, models.NodeInput) for item in encoded)
    assert [item.index for item in encoded] == [1, 2]
    assert encoded[1].node_type == 'image'
    assert encoded[1].text == 'Diagram of the curve.'
    assert not hasattr(encoded[1], 'assets')
    assert not hasattr(encoded[1], 'path')


def test_empty_image_description_uses_local_index():
    nodes = [
        context_window.ContextNode(
            position=3,
            type='image',
            content=None,
        )
    ]

    encoded = pedagogical_component_finder.PedagogicalComponentFinder.encode(
        object(), current_nodes=nodes
    )['current_nodes']

    assert encoded == [models.NodeInput(index=1, node_type='image', text='')]
