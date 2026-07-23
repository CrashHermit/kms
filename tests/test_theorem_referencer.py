"""Theorem referencer: the text it scans (statement + proof) and the in-place `refs` write, around a
scripted LLM pass (no dspy, no network)."""

import asyncio

from kms.core.models import Entity, EntityType, Proof, Reference
from kms.entity.referencers.theorem import reference_text, reference_theorem


class _ScriptedModule:
    def __init__(self, refs):
        self._refs = refs
        self.calls: list[str] = []

    async def references(self, content):
        self.calls.append(content)
        return list(self._refs)


def test_reference_text_includes_statement_and_every_proof():
    entity = Entity(
        type=EntityType.THEOREM,
        members=[0],
        contents=["If $f$ is continuous ..."],
        proofs=[Proof(contents=["By the Mean Value Theorem ..."])],
    )
    assert reference_text(entity) == "If $f$ is continuous ...\n\nBy the Mean Value Theorem ..."


def test_reference_theorem_writes_refs_in_place():
    entity = Entity(
        type=EntityType.THEOREM,
        members=[0],
        contents=["Statement."],
        proofs=[Proof(contents=["Uses Rolle's Theorem."])],
    )
    module = _ScriptedModule([Reference(target="Rolle's Theorem", kind="theorem", tactic="lemma")])
    asyncio.run(reference_theorem(entity, module=module))
    assert [(r.target, r.tactic) for r in entity.refs] == [("Rolle's Theorem", "lemma")]
    assert "Rolle" in module.calls[0]  # the proof text was included in the scan


def test_reference_theorem_skips_the_llm_when_there_is_no_content():
    entity = Entity(type=EntityType.THEOREM, members=[0])
    module = _ScriptedModule([Reference(target="X", kind="theorem", tactic="premise")])
    asyncio.run(reference_theorem(entity, module=module))
    assert entity.refs == []
    assert module.calls == []
