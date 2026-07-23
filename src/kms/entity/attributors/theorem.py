r"""
Theorem attributor — the per-attribute pass over a *found* Theorem entity.

A deliberate copy of the Definition attributor's shape (see ``definition_attributor.py``),
tuned for Theorems and extended with the one thing Theorems add: a **proof**. The Theorem
finder captures a theorem's whole extent — its statement AND its proof — as one flat member
list, leaving statement-vs-proof roles to this pass. So on top of the shared self-contained
attributes we build for Definitions:

    label · number · title · field · contents · bodylist

a Theorem also carries AutoMathKG's Thm-only ``proofs`` (Table B3): a list where each proof
is ``{contents, bodylist}`` here (its own cross-entity ``refs``/``references_tactics`` are
deferred to the graph tier, exactly as for Definitions).

The one structural novelty is splitting the members into statement vs proof. AutoMathKG did
this rule-based on LaTeX environments (``\begin{theorem}`` / ``\begin{proof}``); we live
downstream of OCR with no environment markup, so — as everywhere the paper leaned on LaTeX
structure — we recover it with an LLM: the identity pass returns ``proof_start``, the member
position where the proof begins (``-1`` if none). Unlike the Definition ``label`` boundary,
a wrong split here cannot LOSE content — it only shifts a node between statement and proof,
both of which are kept.

Passes (identity first, then the two bodylists in parallel):

  * ONE LLM CALL — identity: ``label`` + ``number`` + ``title`` + ``field`` + ``proof_start``.
  * DETERMINISTIC — ``contents`` = the STATEMENT members' markdown, label peeled (the proof
    members are held out for ``proofs``).
  * ONE LLM CALL — statement ``bodylist`` over ``STATEMENT_ACTIONS`` (premise / assumption /
    conclusion / enumeration — the hypothesis→conclusion shape; a statement never
    ``definition``s, ``deduction``s, or ``calculation``s).
  * ONE LLM CALL — proof ``bodylist`` over ``PROOF_ACTIONS`` (the reasoning chain: premise,
    assumption, lemma, corollary, deduction, calculation, conclusion, enumeration — the full
    nine minus ``definition``). This is the rich, Lean-4-flavoured decomposition where
    ``bodylist`` actually earns its keep.

Entry point ``attribute_theorem(entity, nodes_by_id)`` (async): writes the attributes onto
the passed Theorem entity and returns it. Persistence-agnostic, like the Definition pass.
"""

import asyncio

import dspy
from pydantic import BaseModel

from kms.core import tracing
from kms.core.llm import text_lm
from kms.core.models import FIELDS, ASTNode, BodySegment, Entity, Proof
from kms.core.state import State

# Subsets of ACTIONS_ALL for the two contexts a theorem's bodylist runs in. A STATEMENT is a
# hypothesis→conclusion assertion (no reasoning steps, no defining); a PROOF is a reasoning
# chain (every role except `definition`). Fewer choices per context, fewer misfires.
STATEMENT_ACTIONS = ["premise", "assumption", "conclusion", "enumeration"]
PROOF_ACTIONS = [
    "premise",
    "assumption",
    "lemma",
    "corollary",
    "deduction",
    "calculation",
    "conclusion",
    "enumeration",
]


class MemberNode(BaseModel):
    """One member node as the identity pass sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class Identify(dspy.Signature):
    r"""
    Read a single mathematical THEOREM — which may be a proposition, corollary, or lemma —
    given as an ordered list of its member nodes, and identify its header information plus
    where its proof begins:

      * label — the theorem's own label as written ("Theorem 3.2", "Lemma 1", "3.2 Theorem",
        "Corollary 4.5"). Empty string if it carries no label.
      * number — just the reference number inside that label ("3.2", "1", "4.5"). Empty if
        there is none.
      * title — a short noun phrase naming the RESULT ("Center of Symmetric Group is Trivial",
        "Pythagoras' Theorem", "Set Union is Associative"). The name of the result, not the
        word "Theorem".
      * field — the single most relevant mathematical field, chosen ONLY from the given list.
      * proof_start — the `position` of the member node where the PROOF begins (usually a node
        that is or starts with "Proof"). The statement is every member before it; the proof is
        that node and everything after. Use -1 if NO proof is shown (all members are statement).
    """

    nodes: list[MemberNode] = dspy.InputField(description="The theorem's member nodes, in order.")
    field_choices: list[str] = dspy.InputField(
        description="The allowed fields; choose exactly one."
    )
    label: str = dspy.OutputField(description="The theorem's label as written, or empty string.")
    number: str = dspy.OutputField(
        description="The reference number in the label, or empty string."
    )
    title: str = dspy.OutputField(description="Short noun phrase naming the result.")
    field: str = dspy.OutputField(description="Exactly one field from the given list.")
    proof_start: int = dspy.OutputField(
        description="Member position where the proof begins, or -1 if no proof is shown."
    )


class StatementBodylist(dspy.Signature):
    r"""
    Segment a THEOREM STATEMENT (the claim itself, NOT its proof) into its logical pieces and
    label each with the role it plays. Return an ordered list of {description, action}.

    THE FOUR ROLES (choose exactly one per piece):
      * premise — a setup clause introducing the objects the statement is about ("Let $n \in
        \mathbb{N}$ be a natural number."). Scaffolding, before the hypotheses.
      * assumption — a hypothesis the theorem imposes: the "if" part, the conditions under
        which the claim holds ("Let $n \ge 3$.", "Suppose $f$ is continuous on $[a,b]$.").
      * conclusion — the claim itself, the "then" part: what the theorem asserts is true
        ("Then the center $Z(S_n)$ of $S_n$ is trivial."). Almost always EXACTLY ONE.
      * enumeration — an itemized or numbered list of hypotheses or cases. The whole list is
        ONE enumeration piece.

    TYPICAL SHAPE: zero or more `premise`/`assumption` clauses giving the hypotheses, then a
    single `conclusion` clause giving the claim. A theorem statement ASSERTS; it does not
    define or prove, so it has no `definition`, `deduction`, or `calculation` pieces.

    PARTITION (critical): the pieces must exactly PARTITION the content — every part belongs
    to EXACTLY ONE piece, no repeats and no omissions; reading the descriptions in order,
    with nothing added or removed, must reproduce the content.

    DESCRIPTIONS: copy each piece's text VERBATIM — reproduce all mathematics and LaTeX
    exactly as given, changing nothing.
    """

    contents: str = dspy.InputField(description="The theorem statement's content (text + LaTeX).")
    actions: list[str] = dspy.InputField(
        description="The allowed action labels for a statement; choose one per piece."
    )
    bodylist: list[BodySegment] = dspy.OutputField(
        description="Ordered {description, action} pieces; descriptions concatenate back to the content."
    )


class ProofBodylist(dspy.Signature):
    r"""
    Segment a mathematical PROOF into its logical steps and label each with the role it plays.
    Return an ordered list of {description, action}.

    THE ROLES (choose exactly one per piece):
      * premise — a setup step fixing objects or notation for the argument ("Let $x \in H$.",
        "Write $n = 2k$.").
      * assumption — a supposition made for the argument ("Assume for contradiction that ...",
        "Suppose $a \ne b$.").
      * lemma — a step that invokes or states an auxiliary lemma used in the argument.
      * corollary — a step that invokes a corollary or immediate consequence.
      * deduction — a logical inference: deriving a fact from previous ones ("Hence ...",
        "It follows that ...", "Therefore $x = y$.").
      * calculation — a computational or algebraic step (manipulating, evaluating, simplifying
        an expression).
      * enumeration — a case split or itemized list of cases ("Case 1: ...", "Case 2: ...").
      * conclusion — the final step that establishes the claim and closes the proof ("This
        proves the theorem.", "as required", a QED / $\square$). Usually EXACTLY ONE, at the end.

    A proof never DEFINES a concept, so there is no `definition` role.

    PARTITION (critical): the pieces must exactly PARTITION the content — every part belongs
    to EXACTLY ONE piece, no repeats and no omissions; reading the descriptions in order,
    with nothing added or removed, must reproduce the content.

    DESCRIPTIONS: copy each piece's text VERBATIM — reproduce all mathematics and LaTeX
    exactly as given, changing nothing.
    """

    contents: str = dspy.InputField(description="The proof's content (text + LaTeX).")
    actions: list[str] = dspy.InputField(
        description="The allowed action labels for a proof; choose one per piece."
    )
    bodylist: list[BodySegment] = dspy.OutputField(
        description="Ordered {description, action} pieces; descriptions concatenate back to the content."
    )


class Identity(BaseModel):
    """The identity pass's result for one theorem."""

    label: str | None = None
    number: str | None = None
    title: str | None = None
    field: str | None = None
    proof_start: int = -1


class Module(dspy.Module):
    """Runs the identity pass and the two bodylist passes (statement, proof) for one theorem."""

    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.identify = dspy.Predict(Identify)
        self.statement_bodylist = dspy.ChainOfThought(StatementBodylist)
        self.proof_bodylist = dspy.ChainOfThought(ProofBodylist)
        self.set_lm(lm or text_lm())

    async def identity(self, members: list[ASTNode]) -> Identity:
        nodes = [
            MemberNode(position=k, type=(m.type.value if m.type else ""), content=m.content)
            for k, m in enumerate(members)
        ]
        r = await self.identify.acall(nodes=nodes, field_choices=FIELDS)
        tracing.record(
            "theorem_identify",
            inputs={"nodes": [n.model_dump() for n in nodes], "field_choices": FIELDS},
            outputs={
                "label": r.label,
                "number": r.number,
                "title": r.title,
                "field": r.field,
                "proof_start": r.proof_start,
            },
        )
        return Identity(
            label=(r.label or None),
            number=(r.number or None),
            title=(r.title or None),
            field=(r.field if r.field in FIELDS else None),
            proof_start=(r.proof_start if isinstance(r.proof_start, int) else -1),
        )

    async def statement_body(self, contents: str) -> list[BodySegment]:
        result = await self.statement_bodylist.acall(contents=contents, actions=STATEMENT_ACTIONS)
        bodylist = [s for s in (result.bodylist or []) if s.action in STATEMENT_ACTIONS]
        tracing.record(
            "theorem_statement_bodylist",
            inputs={"contents": contents, "actions": STATEMENT_ACTIONS},
            outputs={"bodylist": [s.model_dump() for s in bodylist]},
        )
        return bodylist

    async def proof_body(self, contents: str) -> list[BodySegment]:
        result = await self.proof_bodylist.acall(contents=contents, actions=PROOF_ACTIONS)
        bodylist = [s for s in (result.bodylist or []) if s.action in PROOF_ACTIONS]
        tracing.record(
            "theorem_proof_bodylist",
            inputs={"contents": contents, "actions": PROOF_ACTIONS},
            outputs={"bodylist": [s.model_dump() for s in bodylist]},
        )
        return bodylist


def _members(entity: Entity, nodes_by_id: dict[int, ASTNode]) -> list[ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


def _contents(members: list[ASTNode], label: str | None) -> list[str]:
    """The content members as a list of sequence strings, with `label` peeled off the front.

    A standalone label node ("Theorem 3.2") strips to empty and is dropped; a fused label
    ("Theorem 3.2. Let ...") leaves its statement, which is kept; a content-bearing node is
    never dropped wholesale. Passing ``label=None`` (as for the proof half) peels nothing."""
    texts = [m.content for m in members if m.content and m.content.strip()]
    if texts and label:  # peel the label off the first content piece; drop it if that empties it
        head = _strip_label_prefix(texts[0], label)
        texts = ([head] if head.strip() else []) + texts[1:]
    return texts


def _strip_label_prefix(text: str, label: str | None) -> str:
    """Remove a fused label from the front of the first content string, keyed on the
    LLM-extracted label via a plain prefix match — no regex. Unchanged if it does not
    start with the label."""
    if not label or not text:
        return text
    body = text.lstrip()
    lab = label.strip().rstrip(".")
    if lab and body[: len(lab)].lower() == lab.lower():
        return body[len(lab) :].lstrip(" .:\t\n")
    return text


async def attribute_theorem(
    entity: Entity,
    nodes_by_id: dict[int, ASTNode],
    module: Module | None = None,
) -> Entity:
    """Fill in the self-contained attributes on one Theorem entity, in place.

    The identity pass gives label/number/title/field and the ``proof_start`` boundary; the
    members split into statement (before the boundary) and proof (from it) — both halves are
    always kept, so a wrong boundary never loses content. ``contents`` is the label-peeled
    statement; the statement and proof bodylists run in parallel over their own role sets.
    Persistence-agnostic: the enriched entity is returned; where it lands is the caller's concern.
    """
    module = module or Module()
    members = _members(entity, nodes_by_id)
    ident = await module.identity(members)

    ps = ident.proof_start
    has_proof = 0 < ps < len(members)
    statement_members = members[:ps] if has_proof else members
    proof_members = members[ps:] if has_proof else []

    contents = _contents(statement_members, ident.label)
    statement_blob = "\n\n".join(contents)

    async def _statement() -> list[BodySegment]:
        return await module.statement_body(statement_blob) if statement_blob else []

    async def _proof() -> Proof | None:
        if not proof_members:
            return None
        p_contents = _contents(proof_members, None)  # the "Proof." marker is kept for now
        p_blob = "\n\n".join(p_contents)
        p_bodylist = await module.proof_body(p_blob) if p_blob else []
        return Proof(contents=p_contents, bodylist=p_bodylist)

    statement_bodylist, proof = await asyncio.gather(_statement(), _proof())

    entity.label = ident.label
    entity.number = ident.number
    entity.title = ident.title
    entity.field = ident.field
    entity.contents = contents
    entity.bodylist = statement_bodylist
    entity.proofs = [proof] if proof else []
    return entity


# --- LangGraph node: enrich the found Theorems with their attributes ---


class TheoremAttributorNode:
    """Fills in each found Theorem's self-contained attributes (incl. its proof), in place.

    Runs after the Theorem finder, over the ``theorem_entities`` channel it produced. The
    per-entity attributions are independent, so they run concurrently; the enriched entities
    (mutated in place) are written back to the same channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        nodes_by_id = {n.id: n for n in state.get("nodes", []) if n.id is not None}
        entities = state.get("theorem_entities", [])
        if entities:
            await asyncio.gather(
                *(attribute_theorem(e, nodes_by_id, self.module) for e in entities)
            )
        return {"theorem_entities": entities}
