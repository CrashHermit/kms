r"""
Open-relation referencer — the cross-entity pass over an *attributed* entity.

The finders and attributors build a self-contained entity (`contents`, `procedures`, …). This stage
adds the one cross-entity attribute AutoMathKG defines: `refs` — what the entity cites in its
statement or its worked part, each with the relation it stands in. AutoMathKG's own example is a
problem whose solution applies a definition: `{"definition:positive definite matrix": "deduction"}`.

**This replaces the three per-type referencers** (problem / definition / theorem), and the collapse is
what opening the vocabulary makes possible. Those three differed only in the noun their prompt used
and in the closed lists they enforced — `REFERENCE_KINDS` (definition | theorem) for what a target
could be, and the nine `ACTIONS_ALL` tactics for the relation. Both lists are math's, and neither
survives contact with a physics or biology textbook: a law is not a definition or a theorem, and
"conserves" is not one of nine math tactics. So the target kind and the relation both become **open
and LLM-named** (AutoSchemaKG's open-relation model, `docs/GENERALIZATION.md` step 2) — and once
nothing is type-specific, one referencer serves every entity, and every channel.

ONE LLM CALL over the entity's statement AND its worked part (references appear in both). The output
is a list of `models.Reference(target, kind, relation)`; the graph tier resolves each `target` to a
canonical hub keyed by (kind, normalized name), so references from any book/entity converge on one
target. Blank targets are dropped — the only validation left, now that neither vocabulary is closed.

The entry point is `reference(entity, module)` (async): it writes `entity.refs` in place and returns
the entity. Persistence-agnostic — turning `refs` into edges is the entity persister's job.
"""

import asyncio

import dspy

from kms.core import llm, models, state


class ExtractReferences(dspy.Signature):
    r"""
    Read a single textbook block — its statement AND its worked part (proof, solution, derivation) —
    and list the named things it REFERENCES: the definitions, theorems, laws, principles, models, or
    named results it applies, relies on, or builds upon. For each reference give:

      * target — the referenced thing's name as written ("Positive Definite Matrix", "Pythagoras's
        Theorem", "Second Law of Thermodynamics", "Prime Number"). The NAME of what is referenced,
        not a whole sentence.
      * kind — what sort of thing the target is, in ONE lowercase word from the subject's own
        vocabulary: "definition", "theorem" (use it for a proposition, lemma, or corollary), and
        outside mathematics "law", "principle", "model", "mechanism", … There is NO fixed list.
      * relation — how THIS block relates to the target, as a short lowercase verb phrase: "applies",
        "depends on", "generalizes", "specializes", "assumes", "is defined in terms of", "follows
        from", "contradicts". Name the relation that actually holds; there is NO fixed list. Prefer
        the most specific phrase that is accurate.

    Only list genuine references to NAMED things. A reference to a numbered result the block cites
    ("by Theorem 3.7") counts — use the name it is given. Ordinary vocabulary and notation do not.
    If the block references nothing, return an empty list. Do NOT invent references.
    """

    content: str = dspy.InputField(
        description="The block's statement followed by its worked part (text + LaTeX)."
    )
    references: list[models.Reference] = dspy.OutputField(
        description='The referenced things, each with its kind and the relation that holds.'
    )


class Module(dspy.Module):
    """Runs the single open-relation extraction pass for one entity."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractReferences)
        self.set_lm(language_model or llm.text_lm())

    async def references(self, content: str) -> list[models.Reference]:
        """Extracts what this entity's content cites, each with its kind and open relation."""
        result = await self.extract.acall(content=content)
        return [
            models.Reference(
                target=' '.join(ref.target.split()),
                kind=_token(ref.kind),
                relation=_token(ref.relation),
            )
            for ref in (result.references or [])
            if (ref.target or '').strip()
        ]


def _token(value: str | None) -> str:
    """A vocabulary value as a lowercase, whitespace-collapsed token. Open does not mean unruly: the
    kind keys the canonical hub's identity uuid, so "Definition" and "definition " must not mint two
    hubs for one target."""
    return ' '.join((value or '').split()).lower()


def reference_text(entity: models.Entity) -> str:
    """The text an entity's references are drawn from: its statement (`contents`) plus every
    procedure's content — references live in both, and a solution that applies a definition is
    exactly the case AutoMathKG's example is about. Empty when the entity has no content yet."""
    parts = list(entity.contents)
    for procedure in entity.procedures:
        parts.extend(procedure.contents)
    return '\n\n'.join(part for part in parts if part and part.strip())


async def reference(
    entity: models.Entity, module: Module | None = None
) -> models.Entity:
    """Fill `entity.refs` on one entity, in place, from its statement + worked part. A no-op (empty
    refs) when the entity has no content. Returns the same entity."""
    module = module or Module()
    blob = reference_text(entity)
    entity.refs = await module.references(blob) if blob else []
    return entity


class ReferencerNode:
    """Adds each attributed entity's cross-entity `refs`, in place, on one entity channel.

    Runs after that channel's attributor (it needs `contents`). The channel is a constructor
    argument because the same pass serves every chain — the three per-type channels and the block
    channel alike — which is the point of an open vocabulary: nothing here is type-specific. The
    per-entity passes are independent, so they run concurrently; the enriched entities are written
    back to the same channel."""

    def __init__(
        self,
        channel: str = 'block_entities',
        module: Module | None = None,
    ) -> None:
        self.channel = channel
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Extracts each entity's cross-entity references on this node's channel, in place."""
        entities = state.get(self.channel, [])
        if entities:
            await asyncio.gather(
                *(reference(entity, self.module) for entity in entities)
            )
        return {self.channel: entities}
