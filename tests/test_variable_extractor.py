"""Variable extractor — per-node iteration and context assembly.
No network/LLM."""

import asyncio

from kms.core import models, walker
from kms.ingestion import variable_extractor


def _paragraph(content, node_id=0):
    return models.ParagraphNode(content=content, id=node_id)


def _math(content, node_id=0):
    return models.MathNode(content=content, id=node_id)


# --- extract_equations_and_variables (using scripted modules) --------------


class _ScriptedRouter:
    """Returns a fixed (has_equation, has_variable) for every node."""

    def __init__(self, *flags):
        self._flags = list(flags)

    async def aforward(self, content, content_before=None, content_after=None):
        return self._flags.pop(0)


class _ScriptedEquation:
    """Returns a fixed list of equations per call."""

    def __init__(self, *eq_lists):
        self._eq_lists = list(eq_lists)

    async def aforward(self, content, content_before=None, content_after=None):
        return list(self._eq_lists.pop(0))


class _ScriptedVariable:
    """Returns a fixed list of variables per call."""

    def __init__(self, *var_lists):
        self._var_lists = list(var_lists)

    async def aforward(
        self,
        content,
        content_before=None,
        content_after=None,
        equations=None,
    ):
        return list(self._var_lists.pop(0))


def test_iterates_every_node_with_content():
    nodes = [
        _paragraph('First.', node_id=0),
        _paragraph('Second.', node_id=1),
    ]
    router = _ScriptedRouter((True, False), (False, True))
    eq_mod = _ScriptedEquation(
        [models.Equation(latex='$$x=1$$')],
        [],  # second node: no equations
    )
    var_mod = _ScriptedVariable(
        [models.Variable(symbol='y', meaning='why', kind='variable')],
    )
    eqs, vars_ = asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=router,
            equation_module=eq_mod,
            variable_module=var_mod,
        )
    )
    assert eqs == [(0, [models.Equation(latex='$$x=1$$')])]
    assert vars_ == [
        (1, [models.Variable(symbol='y', meaning='why', kind='variable')])
    ]


def test_skips_nodes_without_content():
    nodes = [
        _paragraph('', node_id=0),
        _paragraph('Has content.', node_id=1),
    ]
    router = _ScriptedRouter((True, False))
    eq_mod = _ScriptedEquation(
        [models.Equation(latex='$$z=3$$')],
    )
    var_mod = _ScriptedVariable([])
    eqs, vars_ = asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=router,
            equation_module=eq_mod,
            variable_module=var_mod,
        )
    )
    # Node 0 skipped (empty content), only node 1 processed.
    assert eqs == [(1, [models.Equation(latex='$$z=3$$')])]
    assert vars_ == []


def test_skips_nodes_without_ids():
    nodes = [
        models.ParagraphNode(content='no id'),
    ]
    eqs, vars_ = asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=_ScriptedRouter(),
            equation_module=_ScriptedEquation(),
            variable_module=_ScriptedVariable(),
        )
    )
    assert eqs == []
    assert vars_ == []


def test_router_gates_both_extractors():
    # has_equation=True, has_variable=True → both extractors run.
    nodes = [_paragraph('Both.', node_id=0)]
    router = _ScriptedRouter((True, True))
    eq_mod = _ScriptedEquation(
        [models.Equation(latex='$$e=mc^2$$')],
    )
    var_mod = _ScriptedVariable(
        [models.Variable(symbol='m', meaning='mass', kind='variable')],
    )
    eqs, vars_ = asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=router,
            equation_module=eq_mod,
            variable_module=var_mod,
        )
    )
    assert len(eqs) == 1
    assert len(vars_) == 1


def test_router_gates_neither_with_both_false():
    nodes = [_paragraph('Nothing.', node_id=0)]
    router = _ScriptedRouter((False, False))
    # These would be called but the router says no for both.
    eqs, vars_ = asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=router,
            equation_module=_ScriptedEquation(),
            variable_module=_ScriptedVariable(),
        )
    )
    assert eqs == []
    assert vars_ == []


# --- concurrency ------------------------------------------------------------


def test_results_stay_in_document_order_when_nodes_run_concurrently():
    # Nodes finish out of order (the first sleeps longest), but the channels
    # must still read in document order — the persister and the overlay key
    # off node ids, and a reordered channel makes runs non-reproducible.
    nodes = [_paragraph(f'Node {i}.', node_id=i) for i in range(5)]

    class _SlowestFirstVariable:
        async def aforward(
            self,
            content,
            content_before=None,
            content_after=None,
            equations=None,
        ):
            index = int(content.split()[1].rstrip('.'))
            await asyncio.sleep((5 - index) * 0.01)
            return [
                models.Variable(
                    symbol=f's{index}', meaning='m', kind='variable'
                )
            ]

    class _AlwaysVariable:
        async def aforward(
            self, content, content_before=None, content_after=None
        ):
            return (False, True)

    _, vars_ = asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=_AlwaysVariable(),
            equation_module=_ScriptedEquation(),
            variable_module=_SlowestFirstVariable(),
        )
    )
    assert [node_id for node_id, _ in vars_] == [0, 1, 2, 3, 4]
    assert [b[0].symbol for _, b in vars_] == ['s0', 's1', 's2', 's3', 's4']


def test_max_concurrency_bounds_nodes_in_flight():
    nodes = [_paragraph(f'Node {i}.', node_id=i) for i in range(12)]
    live = 0
    peak = 0

    class _TrackingRouter:
        async def aforward(
            self, content, content_before=None, content_after=None
        ):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return (False, False)

    asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=_TrackingRouter(),
            equation_module=_ScriptedEquation(),
            variable_module=_ScriptedVariable(),
            max_concurrency=3,
        )
    )
    assert peak <= 3
    # And it really did overlap — a serial walk would peak at 1.
    assert peak > 1


def test_equations_feed_into_variable_extractor():
    # The equation output from a node is passed as context to the variable
    # extractor on the same node.
    nodes = [_paragraph('Eq + var.', node_id=0)]
    router = _ScriptedRouter((True, True))
    eq_mod = _ScriptedEquation(
        [models.Equation(latex='$$F=ma$$', name="Newton's second law")],
    )

    seen_equations = []

    class _RecordingVariable:
        async def aforward(
            self,
            content,
            content_before=None,
            content_after=None,
            equations=None,
        ):
            seen_equations.append(equations)
            return []

    asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=router,
            equation_module=eq_mod,
            variable_module=_RecordingVariable(),
        )
    )
    assert len(seen_equations) == 1
    assert seen_equations[0] is not None
    assert seen_equations[0][0].name == "Newton's second law"


def test_context_walking_is_per_node():
    # Each node gets its own before/after context from the node stream.
    nodes = [
        _paragraph('First.', node_id=0),
        _paragraph('Focus.', node_id=1),
        _paragraph('After.', node_id=2),
    ]

    seen_contexts = []

    class _CapturingRouter:
        async def aforward(
            self, content, content_before=None, content_after=None
        ):
            seen_contexts.append((content_before, content_after))
            return True, False

    asyncio.run(
        variable_extractor.extract_equations_and_variables(
            nodes,
            router_module=_CapturingRouter(),
            equation_module=_ScriptedEquation(
                [],
                [],
                [],
            ),
            variable_module=_ScriptedVariable(),
        )
    )
    # All three nodes have content, so three calls.
    assert len(seen_contexts) == 3
    # Node 1 (focus) should see node 0 before and node 2 after.
    assert seen_contexts[1][0] == 'First.'
    assert seen_contexts[1][1] == 'After.'


# --- context_around with renderable list (walker tests, unchanged) ---------


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
