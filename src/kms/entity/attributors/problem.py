r"""
Problem attributor — the per-attribute pass over a *found* Problem entity.

The simplest of the three attributors. A Problem (worked example or exercise) carries only
AutoMathKG's self-contained header attributes plus its solution(s):

    label · number · title · field · contents · solutions

Notably a Problem has **no bodylist** anywhere — Table B3 restricts `bodylist` to Thm/Def,
and even a Problem's `solution` stores an empty bodylist. So, unlike the Theorem attributor,
there are no bodylist passes at all: a Solution reduces to just its `contents` (its
cross-entity `refs`/`references_tactics` are deferred to the graph tier). That leaves a
single LLM call (identity) plus deterministic assembly.

The Problem finder captures a worked example's whole extent — its statement AND a shown
solution — as one flat member list. As for the Theorem's proof, we split statement vs
solution with an LLM ``solution_start`` boundary (member position, ``-1`` if no solution is
shown — the typical exercise). Both halves are always kept, so a wrong boundary shifts a
node but never loses content.

Deliberately NOT here: `instruction`. AutoMathKG has no such attribute, and the imperative
of a grouped exercise ("In Exercises 12-18, find ...") lives in a shared lead-in that is not
a member of the individual problem — so `instruction` is a later cross-entity "governor"
pass, not a per-entity attribute. And `solutions` does not need it: extracting a *shown*
solution is a positional split; only the downstream Math-LLM *completion* of a *missing*
solution needs to understand the ask.

Entry point ``attribute_problem(entity, nodes_by_id)`` (async): writes the attributes onto
the passed Problem entity and returns it. Persistence-agnostic, like the other attributors.
"""

import asyncio

import dspy
from pydantic import BaseModel

from kms.core.llm import text_lm
from kms.core.state import FIELDS, ASTNode, Entity, Solution, State


class MemberNode(BaseModel):
    """One member node as the identity pass sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class Identify(dspy.Signature):
    r"""
    Read a single mathematical PROBLEM — a worked example or an exercise — given as an
    ordered list of its member nodes, and identify its header information plus where its
    solution begins:

      * label — the problem's own label as it appears at the very START of the problem
        ("Example 4.1", "Exercise 12", "4.1 Example", "Problem 3"), INCLUDING a bare leading
        reference number carrying no word ("925.", "3.14", "2.1.12"). Read only what LEADS the
        first member node; empty string if it carries no label.
      * number — just the reference number in that LEADING label ("4.1", "12", "3", "925",
        "2.1.12"). This is the problem's OWN number at its start — NEVER a number that appears
        later inside the statement as a cross-reference to another result. In "2.1.12 Prove
        Proposition 2.1.13." the number is 2.1.12, not 2.1.13; in "3.15 ... use Theorem 3.7"
        it is 3.15, not 3.7. Empty if there is none.
      * title — a short noun phrase naming what the problem is about ("Positive Definiteness
        of a Matrix", "Derivative of a Polynomial"). Not the word "Example" or "Exercise".
      * field — the single most relevant mathematical field, chosen ONLY from the given list.
      * solution_start — the `position` of the member node where the SOLUTION/answer begins
        (often a node that is or starts with "Solution"). The problem statement is every
        member before it; the solution is that node and everything after. Use -1 if NO
        solution is shown (all members are the statement — the typical exercise).
    """

    nodes: list[MemberNode] = dspy.InputField(description="The problem's member nodes, in order.")
    field_choices: list[str] = dspy.InputField(
        description="The allowed fields; choose exactly one."
    )
    label: str = dspy.OutputField(description="The problem's label as written, or empty string.")
    number: str = dspy.OutputField(
        description="The problem's own LEADING reference number (never an in-text cross-reference), or empty string."
    )
    title: str = dspy.OutputField(description="Short noun phrase naming what the problem is about.")
    field: str = dspy.OutputField(description="Exactly one field from the given list.")
    solution_start: int = dspy.OutputField(
        description="Member position where the solution begins, or -1 if none shown."
    )


class Identity(BaseModel):
    """The identity pass's result for one problem."""

    label: str | None = None
    number: str | None = None
    title: str | None = None
    field: str | None = None
    solution_start: int = -1


class Module(dspy.Module):
    """Runs the single identity pass for one problem (no bodylist passes)."""

    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.identify = dspy.Predict(Identify)
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
            solution_start=(r.solution_start if isinstance(r.solution_start, int) else -1),
        )


def _members(entity: Entity, nodes_by_id: dict[int, ASTNode]) -> list[ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


def _contents(members: list[ASTNode], label: str | None) -> list[str]:
    """The content members as a list of sequence strings, with `label` peeled off the front.

    A standalone label node ("Example 4.1") strips to empty and is dropped; a fused label
    ("Example 4.1. Find ...") leaves its statement, which is kept; a content-bearing node is
    never dropped wholesale. Passing ``label=None`` (as for the solution half) peels nothing."""
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


async def attribute_problem(
    entity: Entity,
    nodes_by_id: dict[int, ASTNode],
    module: Module | None = None,
) -> Entity:
    """Fill in the self-contained attributes on one Problem entity, in place.

    A single identity pass gives label/number/title/field and the ``solution_start``
    boundary; the members split into statement (before the boundary) and solution (from it) —
    both halves always kept, so a wrong boundary never loses content. ``contents`` is the
    label-peeled statement; ``solutions`` holds the shown solution's contents (empty for a
    plain exercise). No bodylist. Persistence-agnostic: the enriched entity is returned.
    """
    module = module or Module()
    members = _members(entity, nodes_by_id)
    ident = await module.identity(members)

    ss = ident.solution_start
    has_solution = 0 < ss < len(members)
    statement_members = members[:ss] if has_solution else members
    solution_members = members[ss:] if has_solution else []

    contents = _contents(statement_members, ident.label)
    solution_contents = _contents(solution_members, None) if solution_members else []
    solutions = [Solution(contents=solution_contents)] if solution_contents else []

    entity.label = ident.label
    entity.number = ident.number
    entity.title = ident.title
    entity.field = ident.field
    entity.contents = contents
    entity.solutions = solutions
    return entity


# --- LangGraph node: enrich the found Problems with their attributes ---


class ProblemAttributorNode:
    """Fills in each found Problem's self-contained attributes (incl. its solution), in place.

    Runs after the Problem finder, over the ``problem_entities`` channel it produced. The
    per-entity attributions are independent, so they run concurrently; the enriched entities
    (mutated in place) are written back to the same channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        nodes_by_id = {n.id: n for n in state.get("nodes", []) if n.id is not None}
        entities = state.get("problem_entities", [])
        if entities:
            await asyncio.gather(
                *(attribute_problem(e, nodes_by_id, self.module) for e in entities)
            )
        return {"problem_entities": entities}
