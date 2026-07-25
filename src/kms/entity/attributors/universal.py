r"""
Universal attributor — one pass over any found block, whatever kind of block it is.

Replaces the three per-type attributors (problem / definition / theorem) with a single pass, and
carries the one thing the block finder deliberately does not produce: the block's **type**.

    label · number · title · type · contents

**Why the type lives here and not in the finder.** The finder is pure detection — it says where a
block starts and stops — and adding a classification to it would make boundary-finding and typing
share one prompt and one failure. The attributor, on the other hand, is *already* reading the block's
content to pull out its label, number, and title; naming what kind of block it is is one more field
read from the same text, not a new stage (``docs/GENERALIZATION.md``, "Entity layer").

**The type is OPEN.** It is whatever the book calls the thing — definition, theorem, proposition,
lemma, corollary, example, exercise, and outside mathematics law, principle, model, mechanism,
procedure. It is not chosen from a list, because supplying a list is exactly what stops a physics or
biology textbook from being ingested without new vocabulary. It lands as a *property* on a bare
``:Entity``, never as a Neo4j label — an open vocabulary of labels would grow without bound and every
query would have to know it in advance (``kind = label, type = property``).

Deliberately NOT here, and each for a reason:
  * **The statement/derivation split.** A shown proof or solution is extracted by the procedure
    finder, which runs over every entity and asks one direct question — is there something to work
    out, shown or absent? — instead of deriving the answer from a type. So this pass keeps the
    block's whole extent in ``contents``, and the procedure finder takes the worked part out of it.
  * **``field``.** AutoMathKG's seven-value taxonomy is gone; what a block is about now comes from
    the conceptualizer as several induced concepts (``entity/conceptualizer.py``).
  * **``instruction``.** The directive of a grouped exercise lives in a shared lead-in that is not a
    member of the individual block, so it is a later cross-entity pass, not a per-entity attribute.

Entry point ``attribute(entity, nodes_by_id)`` (async): writes the attributes onto the passed entity
and returns it. Persistence-agnostic, like the per-type attributors it replaces.
"""

import asyncio

import dspy
from pydantic import BaseModel

from kms.core import llm, models, state


class MemberNode(BaseModel):
    """One member node as the identity pass sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class Identify(dspy.Signature):
    r"""
    Read a single textbook BLOCK — given as an ordered list of its member nodes — and identify what
    it is and how it is labelled:

      * label — the block's own label as it appears at the very START of the block ("Theorem 3.2",
        "Definition 1.4", "Example 4.1", "Exercise 12", "4.1 Example", "Newton's Second Law"),
        INCLUDING a bare leading reference number carrying no word ("925.", "3.14", "2.1.12"). Read
        only what LEADS the first member node; empty string if it carries no label.
      * number — just the reference number in that LEADING label ("3.2", "1.4", "12", "925",
        "2.1.12"). This is the block's OWN number at its start — NEVER a number that appears later
        inside the body as a cross-reference to another result. In "2.1.12 Prove Proposition 2.1.13."
        the number is 2.1.12, not 2.1.13; in "3.15 ... use Theorem 3.7" it is 3.15, not 3.7. Empty if
        there is none.
      * title — a short noun phrase naming what the block is ABOUT ("Center of Symmetric Group is
        Trivial", "Positive Definite Matrix", "Conservation of Momentum"). The name of the thing, not
        the words "Theorem" or "Example".
      * type — what KIND of block this is, in the book's own vocabulary and in ONE lowercase word:
        "definition", "theorem", "proposition", "lemma", "corollary", "example", "exercise", and
        outside mathematics "law", "principle", "model", "mechanism", "algorithm", … Read it off the
        block's own label when it carries one ("Theorem 3.2" is a theorem). When the label is only a
        number, infer from what the block DOES: it poses a task to solve → "exercise"; it works
        through a task and shows the answer → "example"; it introduces and names a concept →
        "definition"; it asserts a claim that holds → "theorem". There is NO fixed list — use the
        word the book itself would use.

    Do NOT invent a label, number, or title that the block does not have; return an empty string
    instead. The type is the one field that is always answered.
    """

    nodes: list[MemberNode] = dspy.InputField(
        description="The block's member nodes, in order."
    )
    label: str = dspy.OutputField(
        description="The block's label as written, or empty string."
    )
    number: str = dspy.OutputField(
        description="The block's own LEADING reference number (never an in-text cross-reference), or empty string."
    )
    title: str = dspy.OutputField(
        description='Short noun phrase naming what the block is about.'
    )
    type: str = dspy.OutputField(
        description="What kind of block it is, one lowercase word in the book's own vocabulary."
    )


class Identity(BaseModel):
    """The identity pass's result for one block."""

    label: str | None = None
    number: str | None = None
    title: str | None = None
    type: str | None = None


class Module(dspy.Module):
    """Runs the single identity pass for one block."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.identify = dspy.Predict(Identify)
        self.set_lm(language_model or llm.text_lm())

    async def identity(self, members: list[models.ASTNode]) -> Identity:
        """Returns label, number, title, and the induced type for one block."""
        nodes = [
            MemberNode(
                position=k,
                type=(m.type.value if m.type else ''),
                content=m.content,
            )
            for k, m in enumerate(members)
        ]
        result = await self.identify.acall(nodes=nodes)
        return Identity(
            label=(result.label or None),
            number=(result.number or None),
            title=(result.title or None),
            type=_normalize_type(result.type),
        )


def _normalize_type(value: str | None) -> str | None:
    """The induced type as a single lowercase token, or None when the model returned nothing usable.

    Open does not mean unruly: the type keys a property index and (via the canonical hubs) an
    identity uuid, so "Theorem" and "theorem " must not be two types. Multi-word answers keep their
    words, joined by single spaces."""
    name = ' '.join((value or '').split()).lower()
    return name or None


def _members(
    entity: models.Entity, nodes_by_id: dict[int, models.ASTNode]
) -> list[models.ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


def _contents(members: list[models.ASTNode], label: str | None) -> list[str]:
    """The content members as a list of sequence strings, with `label` peeled off the front.

    A standalone label node ("Theorem 3.2") strips to empty and is dropped; a fused label
    ("Theorem 3.2. Let ...") leaves its statement, which is kept; a content-bearing node is never
    dropped wholesale."""
    texts = [m.content for m in members if m.content and m.content.strip()]
    if (
        texts and label
    ):  # peel the label off the first content piece; drop it if that empties it
        head = _strip_label_prefix(texts[0], label)
        texts = ([head] if head.strip() else []) + texts[1:]
    return texts


def _strip_label_prefix(text: str, label: str | None) -> str:
    """Remove a fused label from the front of the first content string, keyed on the LLM-extracted
    label via a plain prefix match — no regex. Unchanged if it does not start with the label."""
    if not label or not text:
        return text
    body = text.lstrip()
    lab = label.strip().rstrip('.')
    if lab and body[: len(lab)].lower() == lab.lower():
        return body[len(lab) :].lstrip(' .:\t\n')
    return text


async def attribute(
    entity: models.Entity,
    nodes_by_id: dict[int, models.ASTNode],
    module: Module | None = None,
) -> models.Entity:
    """Fill in the self-contained attributes on one block entity, in place.

    A single identity pass gives label/number/title/type; ``contents`` is the label-peeled member
    markdown, kept whole — the worked part is separated out later by the procedure finder.
    Persistence-agnostic: the enriched entity is returned.
    """
    module = module or Module()
    members = _members(entity, nodes_by_id)
    ident = await module.identity(members)

    entity.label = ident.label
    entity.number = ident.number
    entity.title = ident.title
    entity.type = ident.type
    entity.contents = _contents(members, ident.label)
    return entity


# --- LangGraph node: enrich the found blocks with their attributes ---


class UniversalAttributorNode:
    """Fills in each found block's self-contained attributes (including its induced type), in place.

    Runs after the block finder, over the ``block_entities`` channel it produced. The per-entity
    attributions are independent, so they run concurrently; the enriched entities (mutated in place)
    are written back to the same channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Fills in each found block's self-contained attributes, in place."""
        nodes_by_id = {
            n.id: n for n in state.get('nodes', []) if n.id is not None
        }
        entities = state.get('block_entities', [])
        if entities:
            await asyncio.gather(
                *(
                    attribute(entity, nodes_by_id, self.module)
                    for entity in entities
                )
            )
        return {'block_entities': entities}
