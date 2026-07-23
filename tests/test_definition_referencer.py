"""Definition referencer: the text it scans and the in-place `refs` write, around a scripted LLM
pass (no dspy, no network). The reference extraction itself is the injected module's job; here we
check the deterministic assembly and orchestration."""

import asyncio

from kms.core.models import Entity, EntityType, Reference
from kms.entity.referencers.definition import reference_definition, reference_text


class _ScriptedModule:
    def __init__(self, refs):
        self._refs = refs
        self.calls: list[str] = []

    async def references(self, content):
        self.calls.append(content)
        return list(self._refs)


def test_reference_text_is_the_definition_contents():
    entity = Entity(
        type=EntityType.DEFINITION, members=[0], contents=["Let $S$ be a set.", "A ..."]
    )
    assert reference_text(entity) == "Let $S$ be a set.\n\nA ..."


def test_reference_definition_writes_refs_in_place():
    entity = Entity(type=EntityType.DEFINITION, members=[0], contents=["A right triangle ..."])
    module = _ScriptedModule([Reference(target="Triangle", kind="definition", tactic="definition")])
    out = asyncio.run(reference_definition(entity, module=module))
    assert out is entity
    assert [(r.target, r.kind, r.tactic) for r in entity.refs] == [
        ("Triangle", "definition", "definition")
    ]
    assert module.calls == ["A right triangle ..."]


def test_reference_definition_skips_the_llm_when_there_is_no_content():
    entity = Entity(type=EntityType.DEFINITION, members=[0])  # no contents yet
    module = _ScriptedModule([Reference(target="X", kind="definition", tactic="premise")])
    asyncio.run(reference_definition(entity, module=module))
    assert entity.refs == []
    assert module.calls == []  # empty blob -> no round-trip
