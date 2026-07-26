"""Instruction distributor: a growing-window walk that stamps a lead-in's shared directive
onto the Problems it governs. The LLM (which judges the governed extent) is injected scripted."""

import asyncio

from kms.core import models
from kms.entity.instruction_distributor import (
    InstructionDistributorNode,
    distribute_instructions,
)


class _ScriptedGovernor:
    """Replays one (instruction, governed_positions) per govern() call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def acall(self, lead_in, following):
        return self._scripted.pop(0) if self._scripted else ('', [])


def _nodes():
    # 0: lead-in (tagged) | 1-3: three governed exercises | 4: a later ungoverned exercise
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='In Exercises 1.23-1.25, find the eigenvalues.',
            id=0,
            segment_index=0,
            role='instruction',
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='1.23 A', id=1, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='1.24 B', id=2, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='1.25 C', id=3, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='1.26 D', id=4, segment_index=0
        ),
    ]


def _problems():
    return [
        models.Entity(
            type='exercise',
            members=[1],
            number='1.23',
            contents=['A'],
        ),
        models.Entity(
            type='exercise',
            members=[2],
            number='1.24',
            contents=['B'],
        ),
        models.Entity(
            type='exercise',
            members=[3],
            number='1.25',
            contents=['C'],
        ),
        models.Entity(
            type='exercise',
            members=[4],
            number='1.26',
            contents=['D'],
        ),
    ]


def test_stamps_the_governed_run_the_llm_returns():
    # LLM governs the first three following problems (positions 0,1,2), a 4th is visible (bounded).
    gov = _ScriptedGovernor([('find the eigenvalues', [0, 1, 2])])
    problems = _problems()
    asyncio.run(distribute_instructions(_nodes(), problems, module=gov))
    assert [p.instruction for p in problems] == [
        'find the eigenvalues',
        'find the eigenvalues',
        'find the eigenvalues',
        None,
    ]


def test_numberless_lead_in_is_governed_by_meaning_not_numbers():
    # "Answer the following." has no range at all; the walk still governs the run the LLM marks.
    nodes = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Answer the following.',
            id=0,
            segment_index=0,
            role='instruction',
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='prove X', id=1, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='prove Y', id=2, segment_index=0
        ),
    ]
    problems = [
        models.Entity(
            type='exercise',
            members=[1],
            number=None,
            contents=['prove X'],
        ),
        models.Entity(
            type='exercise',
            members=[2],
            number=None,
            contents=['prove Y'],
        ),
    ]
    gov = _ScriptedGovernor([('answer the following', [0, 1])])
    asyncio.run(distribute_instructions(nodes, problems, module=gov))
    assert [p.instruction for p in problems] == [
        'answer the following',
        'answer the following',
    ]


def test_governs_nothing_when_the_llm_returns_no_positions():
    gov = _ScriptedGovernor([('', [])])
    problems = _problems()
    asyncio.run(distribute_instructions(_nodes(), problems, module=gov))
    assert all(p.instruction is None for p in problems)


def test_window_grows_when_the_run_reaches_the_edge():
    # Each problem is larger than one budget, so only one fits per window. The run reaches the
    # window edge on each read, forcing the walk to grow (2000 -> 4000 -> 8000) until all three
    # are visible together and it banks. Three govern() calls prove the two growth steps.
    big = 'x' * 9000  # ~2251 tokens, larger than LOOKAHEAD_BUDGET
    nodes = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Do each.',
            id=0,
            segment_index=0,
            role='instruction',
        )
    ]
    problems = []
    for k in range(3):
        nodes.append(
            models.ASTNode(
                type=models.NodeType.LIST,
                content=big,
                id=k + 1,
                segment_index=0,
            )
        )
        problems.append(
            models.Entity(
                type='exercise',
                members=[k + 1],
                number=str(k + 1),
                contents=[big],
            )
        )
    gov = _ScriptedGovernor(
        [('do each', [0]), ('do each', [0]), ('do each', [0, 1, 2])]
    )
    asyncio.run(distribute_instructions(nodes, problems, module=gov))
    assert [p.instruction for p in problems] == [
        'do each',
        'do each',
        'do each',
    ]


def test_a_new_lead_in_bounds_the_previous_one():
    # Two lead-ins: the first governs only up to (not into) the second's problems.
    nodes = [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Group one: do A.',
            id=0,
            segment_index=0,
            role='instruction',
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='1 first', id=1, segment_index=0
        ),
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content='Group two: do B.',
            id=2,
            segment_index=0,
            role='instruction',
        ),
        models.ASTNode(
            type=models.NodeType.LIST, content='2 second', id=3, segment_index=0
        ),
    ]
    problems = [
        models.Entity(
            type='exercise',
            members=[1],
            number='1',
            contents=['first'],
        ),
        models.Entity(
            type='exercise',
            members=[3],
            number='2',
            contents=['second'],
        ),
    ]
    # First lead-in's candidates are only problem 0 (before the 2nd lead-in); second governs its own.
    gov = _ScriptedGovernor([('do A', [0]), ('do B', [0])])
    asyncio.run(distribute_instructions(nodes, problems, module=gov))
    assert [p.instruction for p in problems] == ['do A', 'do B']


def test_node_writes_channel_and_is_a_noop_without_lead_ins():
    nodes = [
        models.ASTNode(
            type=models.NodeType.LIST, content='1.23 A', id=0, segment_index=0
        )
    ]
    problems = [
        models.Entity(
            type='exercise',
            members=[0],
            number='1.23',
            contents=['A'],
        )
    ]
    node = InstructionDistributorNode(module=_ScriptedGovernor([]))
    out = asyncio.run(node.run({'nodes': nodes, 'entities': problems}))
    assert set(out) == {'entities'}
    assert out['entities'][0].instruction is None
