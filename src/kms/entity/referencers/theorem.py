r"""
Theorem referencer — the cross-entity pass over an *attributed* Theorem entity.

The finders and attributors build a self-contained Theorem (`contents`, `bodylist`, `proofs`, …).
This stage adds the one cross-entity attribute AutoMathKG defines: `refs` — the definitions and
theorems this theorem invokes in its statement or proof, each tagged with the tactic role it plays
(Table C4's `Refs` + `References_tactics` templates, fused into one `Reference` record). A proof that
applies a prior lemma or cites a definition is the canonical case, and is exactly where the "logical
or inferential chains" between theorems come from.

ONE LLM CALL over the theorem's statement AND proof text (references appear in both). The output is a
list of `Reference(target, kind, tactic)`; the graph tier resolves each `target` to a general-entity
hub keyed by (kind, normalized name), so references from any book/entity converge on one target. The
tactic set is the full `ACTIONS_ALL` — a reference can play any role (a `lemma` invoked, a `premise`
taken, a `definition` cited) — with invalid `kind`/`tactic` values dropped.

The entry point is `reference_theorem(entity, module)` (async): it writes `entity.refs` in place and
returns the entity. Persistence-agnostic — turning `refs` into edges is the entity persister's job.
"""

import asyncio

import dspy

from kms.core import tracing
from kms.core.llm import text_lm
from kms.core.models import ACTIONS_ALL, REFERENCE_KINDS, Entity, Reference
from kms.core.state import State


class ExtractReferences(dspy.Signature):
    r"""
    Read a single mathematical THEOREM — its statement AND its proof — and list the other named
    mathematical entities it references: the prior definitions and theorems (including propositions,
    lemmas, corollaries) it invokes or applies. For each reference give:

      * target — the referenced entity's name as written ("Mean Value Theorem", "Order of Structure",
        "Set"). The NAME of the referenced result, not a whole sentence.
      * kind — whether the target is a "definition" or a "theorem". Choose exactly one (treat a
        proposition / lemma / corollary as a "theorem").
      * tactic — the role the reference plays in THIS theorem, chosen ONLY from the given list (a
        prior result applied in the proof is often a "lemma" or "deduction"; an object taken as given
        is a "premise"; a cited definition is a "definition").

    Only list genuine references to named mathematical entities. If the theorem references nothing,
    return an empty list. Do NOT invent references, and do NOT list the theorem itself.
    """

    content: str = dspy.InputField(
        description="The theorem's statement followed by its proof (text + LaTeX)."
    )
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
    """Runs the single reference-extraction pass for one theorem."""

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
            "theorem_references",
            inputs={"content": content, "kinds": REFERENCE_KINDS, "tactics": ACTIONS_ALL},
            outputs={"references": [r.model_dump() for r in refs]},
        )
        return refs


def reference_text(entity: Entity) -> str:
    """The text a theorem's references are drawn from: its statement (`contents`) plus every proof's
    content — references live in both. Empty when the theorem has no content yet."""
    parts = list(entity.contents)
    for proof in entity.proofs:
        parts.extend(proof.contents)
    return "\n\n".join(c for c in parts if c and c.strip())


async def reference_theorem(entity: Entity, module: Module | None = None) -> Entity:
    """Fill `entity.refs` on one Theorem, in place, from its statement + proof. A no-op (empty refs)
    when the theorem has no content. Returns the same entity."""
    module = module or Module()
    blob = reference_text(entity)
    entity.refs = await module.references(blob) if blob else []
    return entity


class TheoremReferencerNode:
    """Adds each attributed Theorem's cross-entity `refs`, in place.

    Runs after the Theorem attributor (it needs `contents`/`proofs`), over the `theorem_entities`
    channel. The per-entity passes are independent, so they run concurrently; the enriched entities
    are written back to the same channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        entities = state.get("theorem_entities", [])
        if entities:
            await asyncio.gather(*(reference_theorem(e, self.module) for e in entities))
        return {"theorem_entities": entities}
