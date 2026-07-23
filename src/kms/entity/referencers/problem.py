r"""
Problem referencer — the cross-entity pass over an *attributed* Problem entity.

The finders and attributors build a self-contained Problem (`contents`, `solutions`, …). This stage
adds the one cross-entity attribute AutoMathKG defines: `refs` — the definitions and theorems this
problem invokes in its statement or solution, each tagged with the tactic role it plays (Table C4's
`Refs` + `References_tactics` templates, fused into one `Reference` record). AutoMathKG's own example
is a problem whose solution applies a definition: `{"definition:positive definite matrix": "deduction"}`.

(AutoMathKG restricts the formal edge SET to origins on Definition/Theorem entities, but still
extracts problem references, and a problem that applies a definition is genuinely useful for
retrieval — so this project keeps problem references as edges too.)

ONE LLM CALL over the problem's statement AND solution text (references appear in both). The output
is a list of `Reference(target, kind, tactic)`; the graph tier resolves each `target` to a
general-entity hub keyed by (kind, normalized name), so references from any book/entity converge on
one target. The tactic set is the full `ACTIONS_ALL`, with invalid `kind`/`tactic` values dropped.

The entry point is `reference_problem(entity, module)` (async): it writes `entity.refs` in place and
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
    Read a single mathematical PROBLEM — its statement AND its solution — and list the named
    mathematical entities it references: the definitions and theorems (including propositions, lemmas,
    corollaries) it applies or relies on to solve the problem. For each reference give:

      * target — the referenced entity's name as written ("Positive Definite Matrix",
        "Pythagoras's Theorem", "Prime Number"). The NAME of the referenced result, not a whole
        sentence.
      * kind — whether the target is a "definition" or a "theorem". Choose exactly one (treat a
        proposition / lemma / corollary as a "theorem").
      * tactic — the role the reference plays in THIS problem, chosen ONLY from the given list (a
        definition applied to solve it is often a "deduction" or "definition"; a theorem invoked is a
        "deduction"; an object taken as given is a "premise").

    Only list genuine references to named mathematical entities. If the problem references nothing,
    return an empty list. Do NOT invent references.
    """

    content: str = dspy.InputField(
        description="The problem's statement followed by its solution (text + LaTeX)."
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
    """Runs the single reference-extraction pass for one problem."""

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
            "problem_references",
            inputs={"content": content, "kinds": REFERENCE_KINDS, "tactics": ACTIONS_ALL},
            outputs={"references": [r.model_dump() for r in refs]},
        )
        return refs


def reference_text(entity: Entity) -> str:
    """The text a problem's references are drawn from: its statement (`contents`) plus every
    solution's content — references live in both. Empty when the problem has no content yet."""
    parts = list(entity.contents)
    for solution in entity.solutions:
        parts.extend(solution.contents)
    return "\n\n".join(c for c in parts if c and c.strip())


async def reference_problem(entity: Entity, module: Module | None = None) -> Entity:
    """Fill `entity.refs` on one Problem, in place, from its statement + solution. A no-op (empty
    refs) when the problem has no content. Returns the same entity."""
    module = module or Module()
    blob = reference_text(entity)
    entity.refs = await module.references(blob) if blob else []
    return entity


class ProblemReferencerNode:
    """Adds each attributed Problem's cross-entity `refs`, in place.

    Runs after the Problem attributor (it needs `contents`/`solutions`), over the `problem_entities`
    channel. The per-entity passes are independent, so they run concurrently; the enriched entities
    are written back to the same channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        entities = state.get("problem_entities", [])
        if entities:
            await asyncio.gather(*(reference_problem(e, self.module) for e in entities))
        return {"problem_entities": entities}
