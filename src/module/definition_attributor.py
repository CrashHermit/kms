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

import re

import dspy
from pydantic import BaseModel

from .state import ASTNode, Entity, NodeType, BodySegment
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

# Sentence splitting for the span-based ("units") bodylist path. A prose node is cut into
# sentence units so the model can label sub-node pieces; display math and list/table blocks
# stay whole. Math spans are masked first so a "." inside `$...$` never triggers a cut.
_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _sentence_split(text: str) -> list[str]:
    """Split prose into sentence units without ever cutting inside a `$...$`/`$$...$$` span."""
    spans: list[str] = []

    def _mask(mo: re.Match) -> str:
        spans.append(mo.group(0))
        return f"\x00{len(spans) - 1}\x00"

    masked = _MATH_SPAN.sub(_mask, text)
    out: list[str] = []
    for piece in _SENTENCE_BOUNDARY.split(masked):
        for i, s in enumerate(spans):
            piece = piece.replace(f"\x00{i}\x00", s)
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


class MemberNode(BaseModel):
    """One member node as the identity pass sees it: a local position and its content."""
    position: int
    type: str
    content: str | None = None


class UnitNode(BaseModel):
    """One atomic content unit (a sentence or a whole block) for the span-based bodylist."""
    position: int
    content: str


class BodySpan(BaseModel):
    """A bodylist piece as an inclusive [start, end] range of unit positions, plus its role."""
    start: int
    end: int
    action: str


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
    """

    nodes: list[MemberNode] = dspy.InputField(description="The definition's member nodes, in order.")
    field_choices: list[str] = dspy.InputField(description="The allowed fields; choose exactly one.")
    label: str = dspy.OutputField(description="The definition's label as written, or empty string.")
    number: str = dspy.OutputField(description="The reference number in the label, or empty string.")
    title: str = dspy.OutputField(description="Short noun phrase naming the defined concept.")
    field: str = dspy.OutputField(description="Exactly one field from the given list.")


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
    original order.

    PARTITION (critical): the pieces must exactly PARTITION the content. Every part of the
    content belongs to EXACTLY ONE piece — never place the same text in two pieces (do not
    repeat a sentence under two roles), and never drop any text. Reading the descriptions in
    order, with nothing added and nothing removed, must reproduce the content exactly. If a
    single sentence plays two roles, assign it the single most fitting one; do not emit it
    twice.

    DESCRIPTIONS: copy each piece's text VERBATIM — reproduce all mathematics and LaTeX
    exactly as given, changing nothing.
    """

    contents: str = dspy.InputField(description="The definition's full content (text + LaTeX).")
    actions: list[str] = dspy.InputField(description="The allowed action labels for a definition; choose one per piece.")
    bodylist: list[BodySegment] = dspy.OutputField(
        description="Ordered {description, action} pieces; descriptions concatenate back to the content."
    )


class BodylistSpans(dspy.Signature):
    r"""
    Segment one mathematical DEFINITION into its logical pieces and label each with its role.
    The content is already split into numbered UNITS (sentences and whole blocks). Group
    consecutive units that share a role and return each group as an inclusive [start, end]
    range of unit positions plus one action label — you do NOT rewrite any text.

    THE FOUR ROLES (choose exactly one per piece):
      * premise — a setup clause introducing the objects the definition is built from, before
        the concept is fixed ("Let $S$ be a set."). Scaffolding, not the definition.
      * definition — the clause that FIXES the concept's meaning: it names the thing or says
        what it consists of / is called. A unit that only introduces notation for the newly
        defined object is `definition`, even when phrased as a condition ("If ... then
        denoted $S_n$").
      * assumption — a condition/constraint that RESTRICTS the objects the definition applies
        to, stated inline ("where $n \ge 3$"). A notation remark is NOT an assumption.
      * enumeration — an itemized/numbered list of conditions, axioms, or cases. A whole
        list block is one enumeration piece.

    TYPICAL SHAPE: zero or more `premise` units, then a single `definition` unit/range that
    fixes the concept, optionally followed by `assumption`/`enumeration`. Almost always
    EXACTLY ONE `definition` piece.

    PARTITION (critical): the ranges must COVER EVERY unit exactly once — contiguous, in
    order, non-overlapping, from the first unit to the last. Do not skip a unit and do not
    place a unit in two ranges.
    """

    units: list[UnitNode] = dspy.InputField(description="The definition's content as numbered units, in order.")
    actions: list[str] = dspy.InputField(description="The allowed action labels for a definition; choose one per piece.")
    pieces: list[BodySpan] = dspy.OutputField(
        description="Ordered [start, end, action] ranges of unit positions; together they cover all units exactly once."
    )


class Identity(BaseModel):
    """The identity pass's result for one definition."""
    label: str | None = None
    number: str | None = None
    title: str | None = None
    field: str | None = None


class Module(dspy.Module):
    """Runs the two LLM passes (identity, then bodylist) for one definition."""

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.identify = dspy.Predict(Identify)
        self.bodylist = dspy.ChainOfThought(Bodylist)
        self.spans = dspy.ChainOfThought(BodylistSpans)
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
        )

    async def body(self, contents: str) -> list[BodySegment]:
        result = await self.bodylist.acall(contents=contents, actions=DEFINITION_ACTIONS)
        return [s for s in (result.bodylist or []) if s.action in DEFINITION_ACTIONS]

    async def body_spans(self, units: list[str]) -> list[BodySegment]:
        """Span-based bodylist: the model returns [start, end, action] ranges over the given
        numbered units; descriptions are sliced from the units, so the model never rewrites
        text and coverage is enforced deterministically (every unit lands in exactly one
        piece — see ``_cover``)."""
        if not units:
            return []
        result = await self.spans.acall(
            units=[UnitNode(position=i, content=u) for i, u in enumerate(units)],
            actions=DEFINITION_ACTIONS,
        )
        return _cover(units, list(result.pieces or []))


def _members(entity: Entity, nodes_by_id: dict[int, ASTNode]) -> list[ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


def _content_parts(
    members: list[ASTNode], label: str | None
) -> list[tuple[NodeType | None, str]]:
    """The content members as (type, text) pairs, with the label peeled off the front.

    The label is stripped from the *first* piece only (the finder anchors a definition's
    span at its label). A standalone label node ("1.2 Definition") strips to empty and is
    dropped; a fused label ("Definition 2.1. A sequence ...") leaves its statement, which is
    kept. Crucially a content-bearing node is never dropped wholesale — that once silently
    lost a defining sentence the identity pass mis-flagged as a label. The node type is kept
    so the span path can split prose but leave math/list blocks whole."""
    parts = [
        (m.type, m.content)
        for m in members
        if m.content and m.content.strip()
    ]
    if parts and label:  # peel the label off the first content piece; drop it if that empties it
        typ, text = parts[0]
        head = _strip_label_prefix(text, label)
        parts = ([(typ, head)] if head.strip() else []) + parts[1:]
    return parts


def _units(parts: list[tuple[NodeType | None, str]]) -> list[str]:
    """Atomic units for the span-based bodylist: prose split into sentences, other blocks
    (display math, lists, tables) kept whole."""
    units: list[str] = []
    for typ, content in parts:
        c = (content or "").strip()
        if not c:
            continue
        if typ in (NodeType.PARAGRAPH, None):
            units.extend(_sentence_split(c))
        else:
            units.append(c)
    return units


def _cover(units: list[str], pieces: list[BodySpan]) -> list[BodySegment]:
    """Turn the model's [start, end, action] ranges into a clean partition of the units.

    Coverage is enforced here, not trusted from the model: each unit gets the label of the
    first valid range covering it; any unit left uncovered inherits its neighbour's label
    (forward then backward fill, `definition` as the last resort). Consecutive same-label
    units are then coalesced into segments whose description is their joined text. The result
    always covers every unit exactly once — no duplication, no omission."""
    n = len(units)
    labels: list[str | None] = [None] * n
    for p in sorted(pieces, key=lambda s: s.start):
        if p.action not in DEFINITION_ACTIONS:
            continue
        start = max(0, min(p.start, n - 1))
        end = max(start, min(p.end, n - 1))
        for i in range(start, end + 1):
            if labels[i] is None:
                labels[i] = p.action
    last = None
    for i in range(n):  # forward-fill uncovered units
        last = labels[i] = labels[i] or last
    nxt = None
    for i in range(n - 1, -1, -1):  # back-fill a leading gap
        nxt = labels[i] = labels[i] or nxt
    labels = [lab or "definition" for lab in labels]

    segments: list[BodySegment] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        segments.append(BodySegment(description=" ".join(units[i:j + 1]), action=labels[i]))
        i = j + 1
    return segments


def _strip_label_prefix(text: str, label: str | None) -> str:
    """Remove a *fused* label ("Definition 1.3.") from the front of the first content string.

    The identity pass records the label separately; when it was fused into a prose node
    (no standalone label node to drop) the label text would otherwise sit in `contents`
    twice. Keyed on the LLM-extracted label string via a plain prefix match — no regex.
    Returns the text unchanged if it does not start with the label."""
    if not label or not text:
        return text
    body = text.lstrip()
    lab = label.strip().rstrip(".")
    if lab and body[: len(lab)].lower() == lab.lower():
        return body[len(lab):].lstrip(" .:\t\n")
    return text


async def attribute_definition(
    entity: Entity,
    nodes_by_id: dict[int, ASTNode],
    module: Module | None = None,
    mode: str = "copy",
) -> Entity:
    """Fill in the self-contained attributes on one Definition entity, in place.

    One LLM call identifies label/number/title/field (and points at the label node); the
    content members are assembled deterministically minus that label node; a second LLM
    call builds the bodylist. ``mode`` picks that second call: ``"copy"`` has the model
    write each description verbatim; ``"spans"`` splits the content into numbered units and
    has the model return unit ranges, so the partition is enforced deterministically. The
    attributes are written onto the passed entity (the same entity the finder produced) and
    it is returned. Persistence-agnostic: whether the enriched entity is dumped to JSON or
    loaded into the graph is the caller's concern.
    """
    module = module or Module()
    members = _members(entity, nodes_by_id)
    ident = await module.identity(members)
    parts = _content_parts(members, ident.label)
    contents = [text for _, text in parts]
    if mode == "spans":
        bodylist = await module.body_spans(_units(parts))
    else:
        blob = "\n\n".join(contents)
        bodylist = await module.body(blob) if blob else []

    entity.label = ident.label
    entity.number = ident.number
    entity.title = ident.title
    entity.field = ident.field
    entity.contents = contents
    entity.bodylist = bodylist
    return entity
