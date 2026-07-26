r"""
Role typer — decides, for each span the group finder cut, whether it is a BLOCK or the
worked DERIVATION that resolves one.

One question, asked once per span: is this text *posing or asserting* something (a block —
a definition, a theorem, a worked example's statement, an exercise), or is it *working
something out* (a derivation — a proof, a solution, a calculation)? That is the whole job.

WHY THIS IS ITS OWN STAGE. The group finder used to answer this itself, as a ``role`` on
every span it emitted, which fused a reliable structural task (where do the units start and
stop?) with a softer semantic one (which of these is the working?). That fusion is what let
a book with no ``Solution.`` markers lose its entire procedural spine: the finder read the
absence of a marker as the absence of a derivation and folded the working into the block.
Split out, the boundary walk keeps its reliable job, and this pass sees ONE span at a time
with nothing else to get right.

The distinction is CLOSED and BINARY — exactly two answers — which is what makes it a
different kind of question from ``block_typer``'s open ``type`` (definition / theorem /
law / mechanism / …). Kind is a label, type is a property (``docs/SCHEMA.md``).

Absence stays structural (``docs/SCHEMA.md``, principle 4): this pass never asks "is there
something to work out?" — it only labels the spans it is given. A unit the book never works
out simply produces no procedure-role span, because the finder never cut one.

Entry point ``type_roles(spans, nodes_by_id)`` (async): returns the ``(entities,
procedure_spans)`` pair the rest of the entity layer consumes. Persistence-agnostic.
"""

import asyncio
import logging

import dspy

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)

# The two roles. Closed and binary — NOT the open `type` vocabulary, which block_typer
# induces downstream.
ENTITY_ROLE = 'entity'
PROCEDURE_ROLE = 'procedure'
SPAN_ROLES = [ENTITY_ROLE, PROCEDURE_ROLE]


class Classify(dspy.Signature):
    r"""
    Decide whether one span of textbook text is a BLOCK or a DERIVATION. Answer with exactly
    one word: "entity" or "procedure". This is domain-neutral — the text may come from a
    math, physics, CS, or biology textbook.

    "entity" — a BLOCK. The text POSES or ASSERTS something. It is one of:
    - a DECLARATIVE STATEMENT: a definition, theorem, proposition, lemma, corollary, axiom,
      or a domain's law / model / rule / principle. It says that something IS SO.
    - a POSED TASK: a worked example's statement ("Example 4.2. Compute $\int_0^1 x^2 dx$."),
      or an exercise for the reader ("12. Prove that $\sqrt 2$ is irrational."). It says what
      is to be DONE, without doing it.

    "procedure" — a DERIVATION. The text WORKS SOMETHING OUT, and it belongs to a block
    stated before it: a proof, a solution, a derivation, a worked calculation. Signs of
    working: it substitutes, integrates, factors, splits into cases, applies a named result,
    computes, or concludes ("hence", "therefore", "so we get", "this completes the proof").

    WORKING IS NOT ONLY ALGEBRA. Text that RESOLVES the block before it is a derivation even
    when it manipulates no symbols at all. All of these are "procedure":
    - EXHIBITING an answer: "Note that $y = 0$ is a solution. But another solution is the
      function ..." — it supplies what the block asked for, so it is the solution.
    - ANALYSING the block's own case or figure: "Here both $G_2$ and $G_3$ are subgraphs of
      $G_1$. But only $G_2$ is an induced subgraph, because ..." — it works the posed example
      out in prose.
    - VERIFYING or JUSTIFYING: checking a condition holds, saying why a result follows, or
      explaining why something FAILS ("$G_4$ is NOT a subgraph, because ...").
    The question is never "does it contain equations?" but "does this text ANSWER or WORK OUT
    what came before it?" If yes, it is "procedure".

    A COMPUTATION SESSION IS A DERIVATION. Unlabelled transcript lines and their printed
    output — "sage: f = x^15 + 1", "sage: f.roots()", "[(12, 1), (10, 1), (4, 1)]", a shell or
    REPL session, a table of computed values — are the working of the block above them, so
    they are "procedure". A trailing sentence that comments on that output ("The output above
    lists each root along with its multiplicity.") is part of the same derivation.

    ITS OWN LABEL MAKES IT A BLOCK — CHECK THIS FIRST. If the text OPENS with a label naming
    it as a unit of the book — "Definition 2.5.1", "SAGE Example 2.5.4.", "Theorem 3.4",
    "Lemma 1.2", "Example 6.7", or a bare leading number ("12.", "949", "2.1.12") — the answer
    is "entity", WHATEVER FOLLOWS THAT LABEL. A labelled example that goes on to work itself
    out, or that contains a worked session and its printed output, is STILL a block: books
    label blocks, not derivations. A derivation never carries a block label of its own — it
    either opens with a derivation marker ("Proof.", "Solution.") or is unlabelled text
    continuing from the block before it.

    A LEADING EXERCISE NUMBER COUNTS AS A LABEL EVEN WITH NO PUNCTUATION AFTER IT, and even
    when everything after it is a bare expression with no words. In a problem set the items
    run "949 25 - 7", "952 x + 8", "957 6 · 3 + 5", "963 20 ÷ (4 + 6) · 5" — a number, then
    the thing the reader must evaluate. Every one of those is an EXERCISE, so the answer is
    "entity". Do NOT read the arithmetic as "computing" and call it a derivation: nothing is
    being worked out, the expression is the task itself and no result is shown. An unnumbered
    expression that PRODUCES a result ("= 18", "so $x = 4$") is different — that is working.

    OTHERWISE, THE TEST IS WHAT THE TEXT DOES, NOT HOW IT IS LABELLED. A "Proof." or
    "Solution." marker means "procedure", but MOST derivations in some books carry no marker
    at all — a worked example's solution frequently just runs on from the statement. For
    UNLABELLED text, never answer "entity" only because a marker word is missing. Conversely,
    a block that merely MENTIONS a method ("Use the power rule to compute the following.") is
    still posing a task, so it is "entity".

    Judge the span in front of you on its own terms. If it both states something and then
    works it out, answer "procedure" only if the working is the bulk of it; a statement with
    a trailing clause is still "entity".
    """

    contents: str = dspy.InputField(
        description="The span's text (markdown + LaTeX), in document order."
    )
    role: str = dspy.OutputField(
        description='Exactly one of "entity" (a block that poses or asserts) or "procedure" '
        '(a derivation that works something out).'
    )


class RoleTyper(dspy.Module):
    """Classifies one span as a block or a derivation."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(Classify)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, contents: str) -> str:
        """Returns the span's role, falling back to ``entity`` for an unusable answer."""
        result = await self.classify.acall(contents=contents)
        raw = ' '.join((result.role or '').split()).lower()
        # `entity` is the fallback: it is the far more common role, and a block wrongly
        # demoted to a derivation would be attached to (and hidden under) its neighbour.
        role = raw if raw in SPAN_ROLES else ENTITY_ROLE
        logger.debug(
            'role: %s%s | from %r',
            role,
            '' if raw in SPAN_ROLES else f' (fallback, model said {raw!r})',
            logs.elide(contents),
        )
        return role


def contents_of(span: list[int], nodes_by_id: dict[int, models.ASTNode]) -> str:
    """The span's member content as one string, in member order, blank-line separated."""
    return '\n\n'.join(
        nodes_by_id[i].content
        for i in span
        if i in nodes_by_id
        and nodes_by_id[i].content
        and nodes_by_id[i].content.strip()
    )


async def type_roles(
    spans: list[list[int]],
    nodes_by_id: dict[int, models.ASTNode],
    module: RoleTyper | None = None,
) -> tuple[list[models.Entity], list[list[int]]]:
    """Split the finder's untyped spans into blocks and derivations.

    Each span is classified independently, so the calls run concurrently. Document order is
    preserved on both sides of the split: the finder emits spans in order and this pass
    keeps that order.

    Args:
        spans: The finder's untyped spans, each a list of member node ids.
        nodes_by_id: The full node stream keyed by stable id.
        module: The classifier module. Created fresh if None.

    Returns:
        An ``(entities, procedure_spans)`` pair in document order. Entities carry their
        member node ids and nothing else — the attributes come from ``block_typer`` and
        ``statement_extractor``.
    """
    module = module or RoleTyper()
    if not spans:
        logger.info('role typer: no spans')
        return [], []

    roles = await asyncio.gather(
        *(module.acall(contents_of(span, nodes_by_id)) for span in spans)
    )
    entities: list[models.Entity] = []
    procedure_spans: list[list[int]] = []
    for span, role in zip(spans, roles, strict=True):
        if role == PROCEDURE_ROLE:
            procedure_spans.append(span)
        else:
            entities.append(models.Entity(members=span))

    logger.info(
        'role typer: %d span(s) -> %d block(s), %d derivation(s)',
        len(spans),
        len(entities),
        len(procedure_spans),
    )
    return entities, procedure_spans


# --- LangGraph node: split the untyped spans into blocks and derivations ---


class RoleTyperNode:
    """Classifies each span from the group finder as a block or a derivation.

    Runs directly after the group finder, over the ``spans`` channel it wrote, and produces
    the ``entities`` overlay plus the ``procedure_spans`` the procedure extractor consumes.
    The per-span classifications are independent, so they run concurrently."""

    def __init__(self, module: RoleTyper | None = None) -> None:
        self.module = module or RoleTyper()

    async def run(self, state: state.State) -> dict:
        """Splits the finder's spans into the entity overlay and the procedure spans."""
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        entities, procedure_spans = await type_roles(
            state.get('spans', []), nodes_by_id, self.module
        )
        return {'entities': entities, 'procedure_spans': procedure_spans}
