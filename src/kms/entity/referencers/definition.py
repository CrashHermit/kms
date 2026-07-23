r"""
Definition referencer — the cross-entity pass over an *attributed* Definition entity.

The finders and attributors build a self-contained Definition (`contents`, `bodylist`, …). This
stage adds the one cross-entity attribute AutoMathKG defines: `refs` — the other definitions and
theorems this definition invokes, each tagged with the tactic role it plays (Table C4's `Refs` +
`References_tactics` templates, fused into one `Reference` record). A definition referencing a more
fundamental definition ("a right triangle references the definition of a triangle") is the canonical
case.

ONE LLM CALL over the definition's own content. The output is a list of `Reference(target, kind,
tactic)`; the graph tier resolves each `target` to a general-entity hub keyed by (kind, normalized
name), so references from any book/entity converge on one target. The tactic set is the full
`ACTIONS_ALL` — a reference can play any role — with invalid `kind`/`tactic` values dropped, mirroring
how the attributor filters its bodylist actions.

The entry point is `reference_definition(entity, module)` (async): it writes `entity.refs` in place
and returns the entity. Persistence-agnostic — turning `refs` into edges is the entity persister's job.
"""

import asyncio

import dspy

from kms.core import tracing
from kms.core.llm import text_lm
from kms.core.models import ACTIONS_ALL, REFERENCE_KINDS, Entity, Reference
from kms.core.state import State


class ExtractReferences(dspy.Signature):
    r"""
    Read a single mathematical DEFINITION and list the other named mathematical entities it
    references — the prior definitions and theorems it invokes or builds on. For each reference give:

      * target — the referenced entity's name as written ("Set", "Composition of Mappings",
        "Triangle"). The NAME of the referenced concept, not a whole sentence.
      * kind — whether the target is a "definition" or a "theorem". Choose exactly one.
      * tactic — the role the reference plays in THIS definition, chosen ONLY from the given list
        (e.g. a foundational definition it is built from is usually "definition"; an object it takes
        as given is a "premise").

    Only list genuine references to named mathematical entities. If the definition references nothing,
    return an empty list. Do NOT invent references, and do NOT list the concept being defined itself.
    """

    content: str = dspy.InputField(description="The definition's full content (text + LaTeX).")
    kinds: list[str] = dspy.InputField(
        description="The allowed target kinds; choose one per reference."
    )
    tactics: list[str] = dspy.InputField(
        description="The allowed tactic labels; choose one per reference."
    )
    references: list[Reference] = dspy.OutputField(
        description="The referenced definitions/theorems, each with its kind and tactic."
    )


class Module(dspy.Module):
    """Runs the single reference-extraction pass for one definition."""

    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractReferences)
        self.set_lm(lm or text_lm())

    async def references(self, content: str) -> list[Reference]:
        result = await self.extract.acall(
            content=content, kinds=REFERENCE_KINDS, tactics=ACTIONS_ALL
        )
        refs = [
            r
            for r in (result.references or [])
            if r.kind in REFERENCE_KINDS and r.tactic in ACTIONS_ALL and r.target.strip()
        ]
        tracing.record(
            "definition_references",
            inputs={"content": content, "kinds": REFERENCE_KINDS, "tactics": ACTIONS_ALL},
            outputs={"references": [r.model_dump() for r in refs]},
        )
        return refs


def reference_text(entity: Entity) -> str:
    """The text a definition's references are drawn from: its own content (a definition has no
    proof/solution). Empty when the definition has no content yet."""
    return "\n\n".join(c for c in entity.contents if c and c.strip())


async def reference_definition(entity: Entity, module: Module | None = None) -> Entity:
    """Fill `entity.refs` on one Definition, in place, from its content. A no-op (empty refs) when
    the definition has no content. Returns the same entity."""
    module = module or Module()
    blob = reference_text(entity)
    entity.refs = await module.references(blob) if blob else []
    return entity


class DefinitionReferencerNode:
    """Adds each attributed Definition's cross-entity `refs`, in place.

    Runs after the Definition attributor (it needs `contents`), over the `definition_entities`
    channel. The per-entity passes are independent, so they run concurrently; the enriched entities
    are written back to the same channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        entities = state.get("definition_entities", [])
        if entities:
            await asyncio.gather(*(reference_definition(e, self.module) for e in entities))
        return {"definition_entities": entities}
