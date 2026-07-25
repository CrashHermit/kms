r"""
Conceptualizer — AutoSchemaKG's ``φ`` conceptualization axis over the collected overlay.

This is what replaced AutoMathKG's ``field`` attribute: instead of asking a model to pick one of
seven fixed mathematical fields, it asks what an element is *about* and takes several answers, and it
does so over any domain — a physics entity comes back "second law of thermodynamics, rotational
dynamics", a biology one "biological catalyst, population genetics" (``docs/GENERALIZATION.md``,
Evidence). A closed field taxonomy could not have produced either, which is the whole argument for the
swap.

Two passes, both flat multi-tag, adapted from AutoSchemaKG's schema-induction prompts (their Figs 5/6
in ``docs/autoschemakg/markdown.md`` §B.2) with **the "1–2 word" constraint relaxed** — that limit was
needed by an 8B, and a frontier model gives precise multi-word concepts instead of coarse ones:

* **Entities** — each entity's concepts, with **graph-context enhancement** (§B.2.2): the prompt sees
  the entity's title, type, and the targets it references, so the abstraction is grounded in the
  node's role in the graph rather than in its text alone. The observed failure mode without it is an
  entity conceptualized by its *parts* ("addition, scalar multiplication") rather than by what it is
  *about* ("vector space"), so the prompt says so explicitly.
* **Events** — each procedure step's concepts, the same conceptualization one level down, so the
  procedural half of the graph gets the same handles as the declarative half.

**Multi-granularity comes from the tag list, not from a tree.** Each pass returns several concepts
spanning specific → general ("normal subgroup", "group theory", "abstract algebra"); a coarse tag ends
up shared by many entities and a fine one by few, so the generality gradient falls out of how the
graph is shared instead of needing a ``:BROADER`` hierarchy. Concepts are **born canonical**: the
graph tier keys them on a global uuid over the normalized name, so the same phrase from two books is
one vertex (``graph.concepts``). Merging genuine paraphrases ("linear algebra" vs "vector spaces") is
NOT this stage's job — that is the embedding/fusion tier.

Entry point ``conceptualize(entity, module)`` (async): writes ``entity.concepts`` and each procedure
step's ``concepts`` in place, and returns the entity. Persistence-agnostic, like the attributors.
"""

import asyncio

import dspy

from kms.core import llm, models, state

# How many concept tags to ask for per element. Enough to span specific → general (the paper's own
# examples run to three), few enough that the model stays on what the element is *about*.
CONCEPT_COUNT = 3


class ConceptualizeEntity(dspy.Signature):
    r"""
    Read a textbook entity — a definition, theorem, worked example, law, mechanism — and name the
    CONCEPTS it is an instance of: the abstract categories it belongs to.

    Return several concepts spanning SPECIFIC to GENERAL. For a definition of a normal subgroup:
    "normal subgroup", "group theory", "abstract algebra" — the specific concept it introduces, the
    area it belongs to, and the broad field. For a physics law: "second law of thermodynamics",
    "thermodynamics", "physics". Order them specific first.

    WHAT A CONCEPT IS: what the entity is ABOUT — the thing a reader would file it under, or search
    for to find it. Concepts may be multi-word; be precise rather than short ("kernel of a linear
    map" beats "kernel", which is ambiguous).

    NOT ITS PARTS. Do NOT list the ingredients that appear inside the statement. A vector-space
    definition is about "vector space" and "linear algebra" — NOT "addition", "scalar
    multiplication", "associativity". Those are its components, not what it is an instance of.

    USE THE CONTEXT. The type, title, and referenced targets tell you what the entity is doing in
    the book; use them to disambiguate. An entity about a "kernel" that references "linear map" is
    about the kernel of a linear map, not an integral kernel or a statistical one.

    DOMAIN-NEUTRAL: name the concepts the entity's own subject uses, whatever the subject is. Do not
    force a mathematical vocabulary onto physics or biology content.
    """

    content: str = dspy.InputField(
        description="The entity's content (text + LaTeX)."
    )
    entity_type: str = dspy.InputField(
        description='What kind of block this is (definition / theorem / law / …), if known.'
    )
    title: str = dspy.InputField(
        description='The short name of what the entity is about, if known.'
    )
    referenced: list[str] = dspy.InputField(
        description='Names of the entities this one references — graph context for disambiguation.'
    )
    count: int = dspy.InputField(
        description='About how many concepts to return.'
    )
    concepts: list[str] = dspy.OutputField(
        description='The concepts this entity instantiates, specific first, general last.'
    )


class ConceptualizeEvent(dspy.Signature):
    r"""
    Read ONE step of a worked derivation — a proof step, a solution step, a calculation — and name
    the CONCEPTS it is an instance of: what is being DONE, and what it is being done to.

    Return several concepts spanning SPECIFIC to GENERAL. For "Assume for contradiction that
    $\sqrt{2} = p/q$ in lowest terms": "proof by contradiction", "irrationality proof", "proof
    technique". For "Substitute $u = x^2$ and differentiate": "substitution", "differentiation",
    "calculus".

    Name the technique or the move, not a restatement of the sentence. Concepts may be multi-word.
    The step's role label (premise / deduction / calculation / …) tells you what kind of move it is;
    use it, but be more specific than it.

    DOMAIN-NEUTRAL: a physics derivation step or a biology mechanism step gets its own subject's
    vocabulary.
    """

    step: str = dspy.InputField(description="The step's text (text + LaTeX).")
    action: str = dspy.InputField(
        description="The step's role label (premise / deduction / calculation / …)."
    )
    count: int = dspy.InputField(
        description='About how many concepts to return.'
    )
    concepts: list[str] = dspy.OutputField(
        description='The concepts this step instantiates, specific first, general last.'
    )


class Module(dspy.Module):
    """Runs the entity and event conceptualization passes."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.entity = dspy.ChainOfThought(ConceptualizeEntity)
        self.event = dspy.ChainOfThought(ConceptualizeEvent)
        self.set_lm(language_model or llm.text_lm())

    async def entity_concepts(
        self,
        content: str,
        entity_type: str,
        title: str,
        referenced: list[str],
    ) -> list[str]:
        """Returns the concepts one entity instantiates, graph-context enhanced."""
        result = await self.entity.acall(
            content=content,
            entity_type=entity_type,
            title=title,
            referenced=referenced,
            count=CONCEPT_COUNT,
        )
        return _clean(result.concepts)

    async def event_concepts(self, step: str, action: str) -> list[str]:
        """Returns the concepts one procedure step instantiates."""
        result = await self.event.acall(
            step=step, action=action, count=CONCEPT_COUNT
        )
        return _clean(result.concepts)


def _clean(concepts: list[str] | None) -> list[str]:
    """The usable concept phrases in a model's answer: non-empty, whitespace-normalized, order
    preserved (specific first), de-duplicated case-insensitively."""
    kept: dict[str, str] = {}
    for concept in concepts or []:
        name = ' '.join((concept or '').split())
        if name and name.lower() not in kept:
            kept[name.lower()] = name
    return list(kept.values())


def entity_text(entity: models.Entity) -> str:
    """The text an entity is conceptualized from: its statement plus every procedure's content. A
    worked example's concepts live as much in how it is solved as in what it asks, so the derivation
    is included — the same reason the referencer reads both halves."""
    parts = list(entity.contents)
    for procedure in entity.procedures:
        parts.extend(procedure.contents)
    return '\n\n'.join(part for part in parts if part and part.strip())


async def conceptualize(
    entity: models.Entity, module: Module | None = None
) -> models.Entity:
    """Fill ``entity.concepts`` and each procedure step's ``concepts``, in place.

    The entity pass and every step pass are independent, so they run concurrently. A no-op (empty
    concepts) for an entity with no content — an entity the attributor never filled has nothing to
    abstract from. Returns the same entity."""
    module = module or Module()
    content = entity_text(entity)
    steps = [
        step for procedure in entity.procedures for step in procedure.steps
    ]

    async def _entity() -> list[str]:
        if not content:
            return []
        return await module.entity_concepts(
            content,
            entity.type or '',
            entity.title or '',
            [ref.target for ref in entity.refs],
        )

    async def _step(step: models.BodySegment) -> list[str]:
        if not (step.description or '').strip():
            return []
        return await module.event_concepts(step.description, step.action)

    concepts, *step_concepts = await asyncio.gather(
        _entity(), *(_step(step) for step in steps)
    )
    entity.concepts = concepts
    for step, tags in zip(steps, step_concepts, strict=True):
        step.concepts = tags
    return entity


class ConceptualizerNode:
    """Tags every collected entity — and every procedure step — with its induced concepts, in place.

    Runs after the collector (it needs the flattened overlay) and before the dependency finder, which
    rolls these concepts up into the prerequisite graph. The per-entity passes are independent, so
    they run concurrently; the enriched entities are written back to the ``entities`` channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Conceptualizes every collected entity and procedure step, in place."""
        entities = state.get('entities', [])
        if entities:
            await asyncio.gather(
                *(conceptualize(entity, self.module) for entity in entities)
            )
        return {'entities': entities}
