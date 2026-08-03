"""Variable extractor — fixed-window iteration, node attribution.
No network/LLM."""

import asyncio

from kms.core import models, walker
from kms.ingestion import variable_extractor


def _paragraph(content, node_id=0):
    return models.ASTNode(type='paragraph', content=content, id=node_id)


def _math(content, node_id=0):
    return models.ASTNode(type='math', content=content, id=node_id)


# --- extract_variables (using scripted modules) -----------------------------


class _ScriptedVariable:
    """Returns a fixed list of (node_id, Variable) pairs per window call."""

    def __init__(self, *var_pairs_lists):
        self._var_pairs_lists = list(var_pairs_lists)

    async def aforward(self, current_nodes):
        return list(self._var_pairs_lists.pop(0))


def test_extracts_from_every_windowed_node():
    nodes = [
        _paragraph('First.', node_id=0),
        _paragraph('Second.', node_id=1),
    ]
    var_mod = _ScriptedVariable(
        [
            (
                1,
                models.Variable(symbol='y', meaning='why', kind='variable'),
            )
        ],
    )
    vars_ = asyncio.run(
        variable_extractor.extract_variables(nodes, variable_module=var_mod)
    )
    assert vars_ == [
        (1, [models.Variable(symbol='y', meaning='why', kind='variable')])
    ]


def test_skips_nodes_without_content():
    nodes = [
        _paragraph('', node_id=0),
        _paragraph('Has content.', node_id=1),
    ]
    var_mod = _ScriptedVariable([])
    vars_ = asyncio.run(
        variable_extractor.extract_variables(nodes, variable_module=var_mod)
    )
    # Node 0 is cut out of every window (empty content); only node 1 reaches
    # the extractor.
    assert vars_ == []


def test_skips_nodes_without_ids():
    nodes = [
        models.ASTNode(type='paragraph', content='no id'),
    ]
    vars_ = asyncio.run(
        variable_extractor.extract_variables(
            nodes,
            variable_module=_ScriptedVariable(),
        )
    )
    assert vars_ == []


def test_large_stream_cuts_into_multiple_windows():
    # 2000-char budget: three fat nodes form three windows, one each, and
    # every window is one call.
    nodes = [_paragraph('a' * 1500, node_id=i) for i in range(3)]

    calls = []

    class _RecordingVariable:
        async def aforward(self, current_nodes):
            calls.append([node.node_id for node in current_nodes])
            return []

    asyncio.run(
        variable_extractor.extract_variables(
            nodes,
            variable_module=_RecordingVariable(),
        )
    )
    assert calls == [[0], [1], [2]]


def test_window_nodes_carry_type_and_content():
    nodes = [
        _math('$$x=1$$', node_id=0),
        _paragraph('Text.', node_id=1),
    ]

    seen = []

    class _RecordingVariable:
        async def aforward(self, current_nodes):
            seen.append([(n.node_id, n.type, n.content) for n in current_nodes])
            return []

    asyncio.run(
        variable_extractor.extract_variables(
            nodes,
            variable_module=_RecordingVariable(),
        )
    )
    assert seen == [[(0, 'math', '$$x=1$$'), (1, 'paragraph', 'Text.')]]


def test_drops_bindings_for_unlisted_node_ids():
    # A binding tagged with a node_id the model was not shown must not mint
    # a channel entry for a phantom node.
    nodes = [_paragraph('Real.', node_id=0)]
    var_mod = _ScriptedVariable(
        [
            (0, models.Variable(symbol='x', meaning='ok', kind='variable')),
            (
                99,
                models.Variable(
                    symbol='ghost', meaning='hallucinated', kind='variable'
                ),
            ),
        ],
    )
    vars_ = asyncio.run(
        variable_extractor.extract_variables(nodes, variable_module=var_mod)
    )
    assert vars_ == [
        (0, [models.Variable(symbol='x', meaning='ok', kind='variable')])
    ]


# --- concurrency ------------------------------------------------------------


def test_results_stay_in_document_order_when_windows_run_concurrently():
    # Windows finish out of order (the first sleeps longest), but the
    # channel must still read in document order — the persister and the
    # overlay key off node ids, and a reordered channel makes runs
    # non-reproducible.
    nodes = [
        _paragraph(f'Node {i}. ' + 'a' * 1400, node_id=i) for i in range(5)
    ]

    class _SlowestFirstVariable:
        async def aforward(self, current_nodes):
            index = int(current_nodes[0].content.split()[1].rstrip('.'))
            await asyncio.sleep((5 - index) * 0.01)
            return [
                (
                    index,
                    models.Variable(
                        symbol=f's{index}', meaning='m', kind='variable'
                    ),
                )
            ]

    vars_ = asyncio.run(
        variable_extractor.extract_variables(
            nodes,
            variable_module=_SlowestFirstVariable(),
        )
    )
    assert [node_id for node_id, _ in vars_] == [0, 1, 2, 3, 4]
    assert [b[0].symbol for _, b in vars_] == ['s0', 's1', 's2', 's3', 's4']


def test_max_concurrency_bounds_windows_in_flight():
    nodes = [_paragraph('a' * 1500, node_id=i) for i in range(12)]
    live = 0
    peak = 0

    class _TrackingVariable:
        async def aforward(self, current_nodes):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return []

    asyncio.run(
        variable_extractor.extract_variables(
            nodes,
            variable_module=_TrackingVariable(),
            max_concurrency=3,
        )
    )
    assert peak <= 3
    # And it really did overlap — a serial walk would peak at 1.
    assert peak > 1


# --- walker helpers (unchanged) --------------------------------------------


def test_fixed_windows_drops_images_and_empty_nodes():
    nodes = [
        models.ASTNode(type='image', content='![1]()', id=0),
        _paragraph('Real.', node_id=1),
        _paragraph('', node_id=2),
    ]
    windows = walker.fixed_windows(nodes, budget=2000)
    assert windows == [[_paragraph('Real.', node_id=1)]]


def test_fixed_windows_cuts_on_budget():
    nodes = [
        _paragraph('a' * 1200, node_id=0),
        _paragraph('b' * 1200, node_id=1),
        _paragraph('c' * 10, node_id=2),
    ]
    windows = walker.fixed_windows(nodes, budget=2000)
    assert [node.id for window in windows for node in window] == [0, 1, 2]
    assert len(windows[0]) == 1
    assert len(windows) == 2


def test_context_around_gives_before_and_after():
    items = [(0, 'Prologue.'), (1, 'Focus.'), (3, 'After.')]
    before, after = walker.context_around(items, cursor=1)
    assert before == 'Prologue.'
    assert after == 'After.'


def test_context_around_at_start_gives_no_before():
    items = [(0, 'First.'), (2, 'Second.')]
    before, after = walker.context_around(items, cursor=0)
    assert before is None
    assert after == 'Second.'


def test_context_around_at_end_gives_no_after():
    items = [(0, 'First.'), (1, 'Last.')]
    before, after = walker.context_around(items, cursor=1)
    assert before == 'First.'
    assert after is None


def test_context_around_respects_token_budgets():
    chunk = 'a' * 300
    items = [
        (0, 'Prologue.'),
        (1, 'Focus.'),
        (2, chunk),
        (3, chunk),
        (4, chunk),
    ]
    before, after = walker.context_around(items, cursor=1)
    assert before == 'Prologue.'
    assert after == f'{chunk}\n\n{chunk}'
