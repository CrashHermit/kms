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

The entry point is ``attribute_definition(entity, nodes_by_id)`` (async); the
``DefinitionAttributorNode`` wrapper at the bottom will wire it into the pipeline once
we are happy with the attributes it produces. Kept persistence-agnostic: it returns a
``DefinitionAttributes`` and says nothing about whether that lands in ``entities.json``
today or a graph vertex tomorrow.
"""

import dspy
from pydantic import BaseModel, Field

from .state import ASTNode, Entity
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

# AutoMathKG's nine role/tactic labels (Table C4, "bodylist" template). Offered in full
# for the first cut; if the model over-reaches for proof-only roles on definitions
# (lemma/corollary/deduction/calculation/conclusion) we can restrict the set later.
ACTIONS = [
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

class MemberNode(BaseModel):
    """One member node as the identity pass sees it: a local position and its content."""
    position: int
    type: str
    content: str | None = None


class BodySegment(BaseModel):
    """One bodylist piece: a slice of the definition's content and its role label."""
    description: str
    action: str


class DefinitionAttributes(BaseModel):
    """The self-contained AutoMathKG Definition attributes this pass fills in."""
    label: str | None = None
    number: str | None = None
    title: str | None = None
    field: str | None = None
    contents: list[str] = Field(default_factory=list)
    bodylist: list[BodySegment] = Field(default_factory=list)


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
    Segment a single mathematical DEFINITION into its logical pieces and label each with
    the role it plays. Return an ordered list of {description, action}.

    SEGMENTS: cut the content where its role changes. A short setup clause ("Let S be a
    set.") is one piece; the clause that actually fixes the meaning is another. A display
    formula or a condition list is its own piece. Keep each piece contiguous and in the
    original order; concatenating the descriptions in order must reproduce the content.

    DESCRIPTIONS: copy the text of each piece VERBATIM — reproduce all mathematics and
    LaTeX exactly as given, changing nothing.

    ACTIONS: label each piece with exactly one role from the given list. For a definition
    the common roles are "premise" (a setup/"let" clause), "definition" (the clause that
    fixes the concept's meaning), "assumption" (a condition imposed), and "enumeration"
    (an itemized list); choose the single most fitting role for each piece.
    """

    contents: str = dspy.InputField(description="The definition's full content (text + LaTeX).")
    actions: list[str] = dspy.InputField(description="The allowed action labels; choose one per piece.")
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
        result = await self.bodylist.acall(contents=contents, actions=ACTIONS)
        return [s for s in (result.bodylist or []) if s.action in ACTIONS]


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
) -> DefinitionAttributes:
    """Fill in the self-contained attributes for one Definition entity.

    One LLM call identifies label/number/title/field (and points at the label node); the
    content members are assembled deterministically minus that label node; a second LLM
    call builds the bodylist. Persistence-agnostic: returns the attributes; the caller
    decides where they land.
    """
    module = module or Module()
    members = _members(entity, nodes_by_id)
    ident = await module.identity(members)
    contents = _contents(members, ident.label_position)
    blob = "\n\n".join(contents)
    bodylist = await module.body(blob) if blob else []

    return DefinitionAttributes(
        label=ident.label,
        number=ident.number,
        title=ident.title,
        field=ident.field,
        contents=contents,
        bodylist=bodylist,
    )
