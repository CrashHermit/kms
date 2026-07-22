"""Problem attributor: the statement/solution split around the single identity pass.

The identity pass (label/number/title/field/solution_start) is injected via a scripted
module, so these tests exercise the real split/assembly logic without dspy."""

import asyncio

from module.state import ASTNode, NodeType, Entity, EntityType
from module.problem_attributor import attribute_problem, Identity


def _nodes():
    return [
        ASTNode(type=NodeType.HEADER, content="Example 4.1", id=0, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Find the derivative of $f(x) = x^2$.", id=1, seg_index=0),
        ASTNode(type=NodeType.PARAGRAPH, content="Solution. $f'(x) = 2x$.", id=2, seg_index=0),
    ]


class _ScriptedModule:
    def __init__(self, identity):
        self._identity = identity

    async def identity(self, members):
        return self._identity


def _run(entity, nodes, module):
    return asyncio.run(attribute_problem(entity, {n.id: n for n in nodes}, module=module))


def test_split_holds_out_solution_and_peels_label():
    nodes = _nodes()
    entity = Entity(type=EntityType.PROBLEM, members=[0, 1, 2])
    ident = Identity(label="Example 4.1", number="4.1", title="Derivative of a monomial",
                     field="analysis", solution_start=2)
    e = _run(entity, nodes, _ScriptedModule(ident))

    assert e.label == "Example 4.1"
    assert e.number == "4.1"
    assert e.field == "analysis"
    # Statement = members before solution_start, label node dropped; solution held out.
    assert e.contents == ["Find the derivative of $f(x) = x^2$."]
    assert e.bodylist == []  # a Problem never has a bodylist
    assert len(e.solutions) == 1
    assert e.solutions[0].contents == ["Solution. $f'(x) = 2x$."]


def test_exercise_with_no_solution_leaves_solutions_empty():
    nodes = _nodes()[:2]  # label + statement, no solution
    entity = Entity(type=EntityType.PROBLEM, members=[0, 1])
    ident = Identity(label="Example 4.1", number="4.1", title="X", field="algebra", solution_start=-1)
    e = _run(entity, nodes, _ScriptedModule(ident))

    assert e.solutions == []
    assert e.contents == ["Find the derivative of $f(x) = x^2$."]


def test_out_of_range_solution_start_is_treated_as_no_solution():
    nodes = _nodes()
    entity = Entity(type=EntityType.PROBLEM, members=[0, 1, 2])
    ident = Identity(label="Example 4.1", number="4.1", title="X", field="analysis", solution_start=9)
    e = _run(entity, nodes, _ScriptedModule(ident))

    assert e.solutions == []
    assert len(e.contents) == 2
