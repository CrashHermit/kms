r"""
Definition attributor — the first per-attribute pass over a *found* Definition entity.

The finders produce sparse entities: ``Entity(type=DEFINITION, members=[node ids])`` —
just pointers into the flat node stream. This stage takes one such Definition plus its
member nodes and fills in AutoMathKG's self-contained Definition attributes (Table B3 /
Appendix C), the ones derivable from the definition's own content with no reference to
any other entity:

    label · number · title · field · contents · bodylist

The cross-entity attributes (``refs`` / ``references_tactics``) are deliberately NOT here:
they are edges between entities and belong to the later graph tier, which resolves them
against the whole populated entity set (and shares the same tactic-label machinery).

Two LLM calls plus a deterministic assembly step:

  * ONE LLM CALL — ``label`` + ``number`` + ``title`` + ``field`` (the "identity" pass).
    All four identify the definition's header, so they share one round-trip. Label reading
    is left to the LLM on purpose — a regex over "Definition 2.1" vs Hefferon's "1.2
    Definition" vs an unlabelled definition is brittle and breaks on the next book's
    convention. The pass also POINTS at the label node by its member index (``-1`` if the
    label is fused into prose or absent), so we can drop it from ``contents`` without a
    regex and without the model re-typing the content body.
  * DETERMINISTIC (no LLM): ``contents`` — the member nodes' markdown minus the node the
    identity pass flagged as the label; a list of sequence strings, as AutoMathKG stores it.
  * ONE LLM CALL — ``bodylist``. Segments ``contents`` into ``{description, action}``
    pieces and labels each with one action from AutoMathKG's role set, in a single pass:
    the segment boundary *is* the role transition, so cutting and labelling are one
    decision, not two. First cut is the simplest thing — the model writes each
    ``description`` in directly (a plain copy); we will see how faithfully it reproduces
    the math before reaching for anything cleverer.

The entry point is ``attribute_definition(entity, nodes_by_id)`` (async): it writes the
attributes onto the passed Definition entity (extending what the finder produced) and
returns it. Wiring it into the pipeline as a per-entity pass is the next step, once we
are happy with the attributes it produces. Kept persistence-agnostic — it says nothing
about whether the enriched entity lands in ``entities.json`` today or a graph vertex
tomorrow.
"""

import dspy
from pydantic import BaseModel

from .state import ASTNode, Entity, BodySegment
from .llm import text_lm


# AutoMathKG's fixed field taxonomy (Table C4, "field" template).
FIELDS = [
    "algebra",
    "geometry",
    "analysis",
    "logic",
    "probability and statistics",
    "applied mathematics",
    "foundations of mathematics",
]

# AutoMathKG's nine role/tactic labels (Table C4, "bodylist" template), the full taxonomy
# shared across all entity types.
ACTIONS_ALL = [
    "premise",
    "assumption",
    "lemma",
    "corollary",
    "definition",
    "conclusion",
    "deduction",
    "calculation",
    "enumeration",
]

# The subset a DEFINITION actually exercises. The proof-oriented roles (lemma, corollary,
# deduction, calculation, conclusion) never legitimately apply to a definition, so we offer
# the model only these four. Fewer choices, fewer misfires: with the full nine an early run
# mislabelled a notation remark `assumption` and inverted premise/definition.
DEFINITION_ACTIONS = ["premise", "definition", "assumption", "enumeration"]

class MemberNode(BaseModel):
    """One member node as the identity pass sees it: a local position and its content."""
    position: int
    type: str
    content: str | None = None


class Identify(dspy.Signature):
    r"""
    Read a single mathematical DEFINITION, given as an ordered list of its member nodes,
    and identify its header information:

      * label — the definition's own label exactly as written, if it has one ("Definition
        2.1", "1.2 Definition", "Definition"). Empty string if the definition carries no
        label.
      * number — just the reference number inside that label ("2.1", "1.2"). Empty string
        if there is none.
      * title — a short noun phrase naming the concept being defined ("Symmetric Group",
        "Vector Space", "Positive Definite Matrix"). The NAME of the thing, not a sentence
        and not the word "Definition".
      * field — the single most relevant mathematical field, chosen ONLY from the given
        list. Pick exactly one.
      * label_position — the `position` of the member node that is JUST the label (a node
        whose whole content is the label, e.g. a heading "1.2 Definition"). Use -1 if the
        label is instead fused into a prose node or the definition has no separate label
        node. This lets the label node be dropped from the content; do not return the
        position of a node that also carries the defining statement.
    """

    nodes: list[MemberNode] = dspy.InputField(description="The definition's member nodes, in order.")
    field_choices: list[str] = dspy.InputField(description="The allowed fields; choose exactly one.")
    label: str = dspy.OutputField(description="The definition's label as written, or empty string.")
    number: str = dspy.OutputField(description="The reference number in the label, or empty string.")
    title: str = dspy.OutputField(description="Short noun phrase naming the defined concept.")
    field: str = dspy.OutputField(description="Exactly one field from the given list.")
    label_position: int = dspy.OutputField(description="Position of the pure-label member node, or -1 if none.")


class Bodylist(dspy.Signature):
    r"""
    Segment one mathematical DEFINITION into its logical pieces and label each piece with
    the role it plays. Return an ordered list of {description, action}.

    THE FOUR ROLES (choose exactly one per piece):
      * premise — a setup clause that introduces the objects the definition is built from,
        BEFORE the concept itself is fixed. Usually a "Let ..." sentence ("Let $S$ be a
        set."). Scaffolding, not the definition itself.
      * definition — the clause that actually FIXES the meaning of the concept: it names
        the thing being defined or says what it consists of / is called ("Then
        $(\Gamma,\circ)$ is called the symmetric group.", "A vector space consists of a set
        $V$ along with two operations ..."). A sentence that only introduces notation or an
        abbreviation for the newly defined object is `definition`, even when phrased as a
        condition ("If $S$ has $n$ elements, then $(\Gamma,\circ)$ is often denoted $S_n$").
      * assumption — a condition or constraint that RESTRICTS the objects the definition
        applies to, stated inline rather than as a list ("subject to the conditions that
        ...", "where $n \ge 3$"). A notation/abbreviation remark is NOT an assumption, even
        when it opens with "if".
      * enumeration — an itemized or numbered list of conditions, axioms, or cases (e.g. a
        numbered list of vector-space axioms). The whole list is ONE enumeration piece.

    TYPICAL SHAPE: zero or more `premise` setup clauses, then a single `definition` clause
    that fixes the concept, optionally followed by `assumption`/`enumeration` pieces for its
    conditions. A definition has almost always EXACTLY ONE `definition` piece — the clause
    that says what the concept is or is called. Do not label the core defining clause
    `premise`, and do not split it across pieces.

    SEGMENTS: cut the content where its role changes; keep each piece contiguous and in the
    original order. Concatenating the descriptions in order must reproduce the content.

    DESCRIPTIONS: copy each piece's text VERBATIM — reproduce all mathematics and LaTeX
    exactly as given, changing nothing.
    """

    contents: str = dspy.InputField(description="The definition's full content (text + LaTeX).")
    actions: list[str] = dspy.InputField(description="The allowed action labels for a definition; choose one per piece.")
    bodylist: list[BodySegment] = dspy.OutputField(
        description="Ordered {description, action} pieces; descriptions concatenate back to the content."
    )


class Identity(BaseModel):
    """The identity pass's result for one definition."""
    label: str | None = None
    number: str | None = None
    title: str | None = None
    field: str | None = None
    label_position: int = -1


class Module(dspy.Module):
    """Runs the two LLM passes (identity, then bodylist) for one definition."""

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.identify = dspy.Predict(Identify)
        self.bodylist = dspy.ChainOfThought(Bodylist)
        self.set_lm(lm or text_lm())

    async def identity(self, members: list[ASTNode]) -> Identity:
        nodes = [
            MemberNode(position=k, type=(m.type.value if m.type else ""), content=m.content)
            for k, m in enumerate(members)
        ]
        r = await self.identify.acall(nodes=nodes, field_choices=FIELDS)
        return Identity(
            label=(r.label or None),
            number=(r.number or None),
            title=(r.title or None),
            field=(r.field if r.field in FIELDS else None),
            label_position=(r.label_position if isinstance(r.label_position, int) else -1),
        )

    async def body(self, contents: str) -> list[BodySegment]:
        result = await self.bodylist.acall(contents=contents, actions=DEFINITION_ACTIONS)
        return [s for s in (result.bodylist or []) if s.action in DEFINITION_ACTIONS]


def _members(entity: Entity, nodes_by_id: dict[int, ASTNode]) -> list[ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


def _contents(members: list[ASTNode], label_position: int) -> list[str]:
    """The content members as a list of sequence strings (AutoMathKG's `contents`),
    dropping the node the identity pass flagged as the pure label."""
    return [
        m.content
        for k, m in enumerate(members)
        if k != label_position and m.content and m.content.strip()
    ]


async def attribute_definition(
    entity: Entity,
    nodes_by_id: dict[int, ASTNode],
    module: Module | None = None,
) -> Entity:
    """Fill in the self-contained attributes on one Definition entity, in place.

    One LLM call identifies label/number/title/field (and points at the label node); the
    content members are assembled deterministically minus that label node; a second LLM
    call builds the bodylist. The attributes are written onto the passed entity (the same
    entity the finder produced) and it is returned. Persistence-agnostic: whether the
    enriched entity is dumped to JSON or loaded into the graph is the caller's concern.
    """
    module = module or Module()
    members = _members(entity, nodes_by_id)
    ident = await module.identity(members)
    contents = _contents(members, ident.label_position)
    blob = "\n\n".join(contents)
    bodylist = await module.body(blob) if blob else []

    entity.label = ident.label
    entity.number = ident.number
    entity.title = ident.title
    entity.field = ident.field
    entity.contents = contents
    entity.bodylist = bodylist
    return entity
