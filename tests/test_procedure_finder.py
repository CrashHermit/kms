"""Procedure finder: the four-way routing — extract a shown derivation, create one for a posed task,
defer an unproved claim, skip a block with nothing to work out. The three LLM passes are injected via
a scripted module, so this exercises the routing and the statement/derivation split without dspy."""

import asyncio

from kms.core import models
from kms.entity.procedure_finder import (
    Assessment,
    ProcedureFinderNode,
    find_procedure,
    task_text,
)


class _ScriptedModule:
    """A stand-in Module with a fixed assessment, decomposition, and creation result."""

    def __init__(self, assessment, steps=None, solution=('', [])):
        self._assessment = assessment
        self._steps = steps or []
        self._solution = solution
        self.solved = []

    async def assessment(self, contents):
        return self._assessment

    async def steps(self, contents):
        return list(self._steps)

    async def solve(self, task):
        self.solved.append(task)
        return self._solution


def _run(entity, module):
    return asyncio.run(find_procedure(entity, module))


def test_shown_derivation_is_extracted_and_split_from_the_statement():
    entity = models.Entity(
        type='theorem',
        contents=['The center of $S_n$ is trivial.', 'Proof. Clear.'],
    )
    step = models.BodySegment(description='Clear.', action='conclusion')
    e = _run(
        entity,
        _ScriptedModule(
            Assessment(
                has_work=True, shown=True, start=1, procedure_type='proof'
            ),
            steps=[step],
        ),
    )
    assert e.contents == ['The center of $S_n$ is trivial.']
    assert len(e.procedures) == 1
    assert e.procedures[0].type == 'proof'
    assert e.procedures[0].contents == ['Proof. Clear.']
    assert [s.action for s in e.procedures[0].steps] == ['conclusion']
    assert e.procedures[0].generated is False  # it came off the page


def test_an_absent_derivation_on_a_posed_task_is_created_and_marked_generated():
    entity = models.Entity(type='exercise', contents=['Differentiate $x^2$.'])
    module = _ScriptedModule(
        Assessment(
            has_work=True,
            shown=False,
            start=-1,
            poses_task=True,
            procedure_type='solution',
        ),
        solution=(
            "$f'(x) = 2x$.",
            [
                models.BodySegment(
                    description='Apply the power rule.', action='calculation'
                )
            ],
        ),
    )
    e = _run(entity, module)
    assert e.contents == ['Differentiate $x^2$.']  # the statement is untouched
    assert e.procedures[0].generated is True
    assert e.procedures[0].contents == ["$f'(x) = 2x$."]


def test_an_absent_proof_on_an_asserted_claim_is_deferred():
    # generating proofs is deferred work; the gap stays visible rather than being filled
    entity = models.Entity(type='theorem', contents=['Every field is a ring.'])
    module = _ScriptedModule(
        Assessment(has_work=True, shown=False, start=-1, poses_task=False),
        solution=('should not be used', []),
    )
    e = _run(entity, module)
    assert e.procedures == []
    assert module.solved == []  # the creator was never called


def test_a_block_with_nothing_to_work_out_is_skipped():
    entity = models.Entity(type='definition', contents=['A group is a set.'])
    e = _run(entity, _ScriptedModule(Assessment(has_work=False)))
    assert e.procedures == []
    assert e.contents == ['A group is a set.']


def test_an_unusable_boundary_leaves_the_contents_whole():
    entity = models.Entity(type='theorem', contents=['a', 'b'])
    e = _run(
        entity,
        _ScriptedModule(Assessment(has_work=True, shown=True, start=9)),
    )
    assert e.procedures == [] and e.contents == ['a', 'b']


def test_a_created_solution_is_asked_with_its_shared_instruction():
    # a bare "12. $x^2$" exercise carries its ask in the group lead-in, so completion needs it
    entity = models.Entity(
        type='exercise',
        contents=['$x^2 + 3x + 2$'],
        instruction='Factor each expression.',
    )
    assert task_text(entity) == 'Factor each expression.\n\n$x^2 + 3x + 2$'


def test_node_run_writes_back_to_the_block_channel():
    node = ProcedureFinderNode(
        module=_ScriptedModule(Assessment(has_work=False))
    )
    out = asyncio.run(
        node.run(
            {
                'block_entities': [
                    models.Entity(type='definition', contents=['x'])
                ]
            }
        )
    )
    assert list(out) == ['block_entities']
