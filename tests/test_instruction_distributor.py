"""Instruction distributor — governance hubs, and the verbatim guarantee.

No network/LLM: the governance module is scripted.
"""

import asyncio

from kms.core import models
from kms.ingestion import instruction_distributor


def _lead_in(content, node_id):
    return models.InstructionNode(content=content, id=node_id)


def _exercise(content, node_id):
    return models.ParagraphNode(content=content, id=node_id)


class _ScriptedGovernance:
    """Returns a fixed (instruction, governed_positions) per call."""

    def __init__(self, *answers):
        self._answers = list(answers)
        self.calls = []

    async def aforward(self, lead_in, following):
        self.calls.append((lead_in, [f.content for f in following]))
        return self._answers.pop(0)


def _run(nodes, module):
    return asyncio.run(
        instruction_distributor.distribute_instructions(nodes, module=module)
    )


def test_governed_node_content_is_left_verbatim():
    # The whole point of the hub: a governed exercise is a verbatim page
    # block, so the directive must NOT be written into its content.
    nodes = [
        _lead_in('In the following exercises, simplify.', 0),
        _exercise('979. 17a + 9a', 1),
        _exercise('980. 18z + 9z', 2),
    ]
    module = _ScriptedGovernance(('simplify', [0, 1]))
    cleaned, instructions = _run(nodes, module)

    assert [node.content for node in cleaned] == [
        '979. 17a + 9a',
        '980. 18z + 9z',
    ]
    assert len(instructions) == 1


def test_hub_carries_verbatim_text_directive_and_members():
    nodes = [
        _lead_in('In the following exercises, simplify.', 0),
        _exercise('979. 17a + 9a', 1),
        _exercise('980. 18z + 9z', 2),
    ]
    module = _ScriptedGovernance(('simplify', [0, 1]))
    _, instructions = _run(nodes, module)

    hub = instructions[0]
    # The page's own sentence, not the normalised imperative.
    assert hub.text == 'In the following exercises, simplify.'
    assert hub.directive == 'simplify'
    assert hub.members == [1, 2]
    assert hub.node_id == 0


def test_lead_in_nodes_are_removed_from_the_stream():
    nodes = [
        _lead_in('Lead one.', 0),
        _exercise('1. a', 1),
        _lead_in('Lead two.', 2),
        _exercise('2. b', 3),
    ]
    module = _ScriptedGovernance(('one', [0]), ('two', [0]))
    cleaned, instructions = _run(nodes, module)

    assert all(
        not isinstance(node, models.InstructionNode) for node in cleaned
    )
    assert [node.id for node in cleaned] == [1, 3]
    assert [hub.node_id for hub in instructions] == [0, 2]
    assert [hub.members for hub in instructions] == [[1], [3]]


def test_a_lead_in_governing_nothing_makes_no_hub():
    nodes = [
        _lead_in('Governs nothing here.', 0),
        _exercise('1. a', 1),
    ]
    module = _ScriptedGovernance(('ignored', []))
    cleaned, instructions = _run(nodes, module)

    assert instructions == []
    # The lead-in still leaves the stream.
    assert [node.id for node in cleaned] == [1]


def test_no_lead_ins_is_a_noop_returning_the_stream_unchanged():
    nodes = [_exercise('1. a', 0), _exercise('2. b', 1)]
    cleaned, instructions = _run(nodes, _ScriptedGovernance())

    assert cleaned is nodes
    assert instructions == []


def test_governance_stops_at_the_next_lead_in():
    # The second lead-in's exercises are never candidates for the first.
    nodes = [
        _lead_in('First.', 0),
        _exercise('1. a', 1),
        _lead_in('Second.', 2),
        _exercise('2. b', 3),
    ]
    module = _ScriptedGovernance(('first', [0]), ('second', [0]))
    _run(nodes, module)

    first_call_candidates = module.calls[0][1]
    assert first_call_candidates == ['1. a']
