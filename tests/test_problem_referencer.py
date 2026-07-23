"""Problem referencer: the text it scans (statement + solution) and the in-place `refs` write, around
a scripted LLM pass (no dspy, no network)."""

import asyncio

from kms.core.models import Entity, EntityType, Reference, Solution
from kms.entity.referencers.problem import reference_problem, reference_text


class _ScriptedModule:
    def __init__(self, refs):
        self._refs = refs
        self.calls: list[str] = []

    async def references(self, content):
        self.calls.append(content)
        return list(self._refs)


def test_reference_text_includes_statement_and_every_solution():
    entity = Entity(
        type=EntityType.PROBLEM,
        members=[0],
        contents=["Is $A$ positive definite?"],
        solutions=[Solution(contents=["By the definition of positive definiteness ..."])],
    )
    assert reference_text(entity) == (
        "Is $A$ positive definite?\n\nBy the definition of positive definiteness ..."
    )


def test_reference_problem_writes_refs_in_place():
    entity = Entity(
        type=EntityType.PROBLEM,
        members=[0],
        contents=["Compute."],
        solutions=[Solution(contents=["Apply the positive definite matrix definition."])],
    )
    module = _ScriptedModule(
        [Reference(target="Positive Definite Matrix", kind="definition", tactic="deduction")]
    )
    asyncio.run(reference_problem(entity, module=module))
    assert [(r.target, r.kind, r.tactic) for r in entity.refs] == [
        ("Positive Definite Matrix", "definition", "deduction")
    ]
    assert "positive definite" in module.calls[0]  # solution text included in the scan


def test_reference_problem_skips_the_llm_when_there_is_no_content():
    entity = Entity(type=EntityType.PROBLEM, members=[0])
    module = _ScriptedModule([Reference(target="X", kind="definition", tactic="premise")])
    asyncio.run(reference_problem(entity, module=module))
    assert entity.refs == []
    assert module.calls == []
