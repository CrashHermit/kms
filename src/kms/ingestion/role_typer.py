r"""
Role typer — decides, for each span the group finder cut, whether it is a BLOCK
or the worked DERIVATION that resolves one.

One question, asked once per span: is this text *posing or asserting* something
(a block — a definition, a theorem, a worked example's statement, an exercise),
or is it *working something out* (a derivation — a proof, a solution, a
calculation)? That is the whole job.

WHY THIS IS ITS OWN STAGE. The group finder used to answer this itself, as a
``role`` on every span it emitted, which fused a reliable structural task (where
do the units start and stop?) with a softer semantic one (which of these is the
working?). That fusion is what let a book with no ``Solution.`` markers lose its
entire procedural spine: the finder read the absence of a marker as the absence
of a derivation and folded the working into the block. Split out, the boundary
walk keeps its reliable job, and this pass sees ONE span at a time with nothing
else to get right.

The distinction is CLOSED and BINARY — exactly two answers — which is what makes
it a different kind of question from ``block_typer``'s open ``type`` (definition
/ theorem / law / mechanism / …). Kind is a label, type is a property
(``docs/SCHEMA.md``).

Absence stays structural (``docs/SCHEMA.md``, principle 4): this pass never asks
"is there something to work out?" — it only labels the spans it is given. A unit
the book never works out simply produces no procedure-role span, because the
finder never cut one.

Entry point ``type_roles(spans, nodes_by_id)`` (async): returns the ``(entities,
procedure_ids)`` pair the rest of the pipeline consumes. Persistence-agnostic.
"""

import asyncio
import logging

import dspy

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)

# The two roles. Closed and binary — NOT the open `type` vocabulary, which
# block_typer induces downstream.
STATEMENT_ROLE = 'statement'
PROCEDURE_ROLE = 'procedure'
SPAN_ROLES = [STATEMENT_ROLE, PROCEDURE_ROLE]


class Classify(dspy.Signature):
    r"""
    Decide whether one span of textbook text is a BLOCK or a DERIVATION. Answer
    with exactly one word: "statement" or "procedure". This is domain-neutral —
    the text may come from a math, physics, CS, or biology textbook.

    "statement" — a BLOCK. The text POSES or ASSERTS something. It is one of:
    - a DECLARATIVE STATEMENT: a definition, theorem, proposition, lemma,
      corollary, axiom, or a domain's law / model / rule / principle. It says
      that something IS SO.
    - a POSED TASK: a worked example's statement ("Example 4.2. Compute
      $\int_0^1 x^2 dx$."), or an exercise for the reader ("12. Prove that
      $\sqrt 2$ is irrational."). It says what is to be DONE, without doing it.

    "procedure" — a DERIVATION. The text WORKS SOMETHING OUT, and it belongs to
    a block stated before it: a proof, a solution, a derivation, a worked
    calculation. Signs of working: it substitutes, integrates, factors, splits
    into cases, applies a named result, computes, or concludes ("hence",
    "therefore", "so we get", "this completes the proof").

    WORKING IS NOT ONLY ALGEBRA. Text that RESOLVES the block before it is a
    derivation even when it manipulates no symbols at all. All of these are
    "procedure":
    - EXHIBITING an answer: "Note that $y = 0$ is a solution. But another
      solution is the function ..." — it supplies what the block asked for, so
      it is the solution.
    - ANALYSING the block's own case or figure: "Here both $G_2$ and $G_3$ are
      subgraphs of $G_1$. But only $G_2$ is an induced subgraph, because ..." —
      it works the posed example out in prose.
    - VERIFYING or JUSTIFYING: checking a condition holds, saying why a result
      follows, or explaining why something FAILS ("$G_4$ is NOT a subgraph,
      because ..."). The question is never "does it contain equations?" but
      "does this text ANSWER or WORK OUT what came before it?" If yes, it is
      "procedure".

    A COMPUTATION SESSION IS A DERIVATION. Unlabelled transcript lines and their
    printed output — "sage: f = x^15 + 1", "sage: f.roots()", "[(12, 1), (10,
    1), (4, 1)]", a shell or REPL session, a table of computed values — are the
    working of the block above them, so they are "procedure". A trailing
    sentence that comments on that output ("The output above lists each root
    along with its multiplicity.") is part of the same derivation.

    ITS OWN LABEL MAKES IT A BLOCK — CHECK THIS FIRST. If the text OPENS with a
    label naming it as a unit of the book — "Definition 2.5.1", "SAGE Example
    2.5.4.", "Theorem 3.4", "Lemma 1.2", "Example 6.7", or a bare leading number
    ("12.", "949", "2.1.12") — the answer is "statement", WHATEVER FOLLOWS THAT
    LABEL. A labelled example that goes on to work itself out, or that contains
    a worked session and its printed output, is STILL a block: books label
    blocks, not derivations. A derivation never carries a block label of its own
    — it either opens with a derivation marker ("Proof.", "Solution.") or is
    unlabelled text continuing from the block before it.

    A LEADING EXERCISE NUMBER COUNTS AS A LABEL EVEN WITH NO PUNCTUATION AFTER
    IT, and even when everything after it is a bare expression with no words. In
    a problem set the items run "949 25 - 7", "952 x + 8", "957 6 · 3 + 5", "963
    20 ÷ (4 + 6) · 5" — a number, then the thing the reader must evaluate. Every
    one of those is an EXERCISE, so the answer is "statement". Do NOT read the
    arithmetic as "computing" and call it a derivation: nothing is being worked
    out, the expression is the task itself and no result is shown. An unnumbered
    expression that PRODUCES a result ("= 18", "so $x = 4$") is different — that
    is working.

    OTHERWISE, THE TEST IS WHAT THE TEXT DOES, NOT HOW IT IS LABELLED. A
    "Proof." or "Solution." marker means "procedure", but MOST derivations in
    some books carry no marker at all — a worked example's solution frequently
    just runs on from the statement. For UNLABELLED text, never answer
    "statement" only because a marker word is missing. Conversely, a block that
    merely MENTIONS a method ("Use the power rule to compute the following.") is
    still posing a task, so it is "statement".

    Judge the span in front of you on its own terms. If it both states something
    and then works it out, answer "procedure" only if the working is the bulk of
    it; a statement with a trailing clause is still "statement".
    """

    contents: str = dspy.InputField(
        description="The span's text (markdown + LaTeX), in document order."
    )
    role: str = dspy.OutputField(
        description='Exactly one of "statement" (a statement that poses or asserts) or "procedure" '
        '(a derivation that works something out).'
    )


class RoleTyper(dspy.Module):
    """Classifies one span as a block or a derivation.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(Classify)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, contents: str) -> str:
        """Classify one span.

        Args:
            contents: The span's text, in document order.

        Returns:
            The span's role, falling back to ``statement`` for an unusable
            answer.
        """
        result = await self.classify.acall(contents=contents)
        answer = ' '.join((result.role or '').split()).lower()
        # `statement` is the fallback: it is the far more common role, and a
        # block wrongly demoted to a derivation would be attached to (and
        # hidden under) its neighbour.
        role = answer if answer in SPAN_ROLES else STATEMENT_ROLE
        logger.debug(
            'role: %s%s | from %r',
            role,
            ''
            if answer in SPAN_ROLES
            else f' (fallback, model said {answer!r})',
            logs.elide(contents),
        )
        return role

    def forward(self, contents: str) -> str:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(contents))


def contents_of(span: list[int], nodes_by_id: dict[int, models.ASTNode]) -> str:
    """The span's member content as one blank-line separated string.

    Args:
        span: The span's member node ids, in document order.
        nodes_by_id: The full node stream keyed by stable id.

    Returns:
        The members' content joined in member order.
    """
    return '\n\n'.join(
        nodes_by_id[node_id].content
        for node_id in span
        if node_id in nodes_by_id
        and nodes_by_id[node_id].content
        and nodes_by_id[node_id].content.strip()
    )


def _mark_statement(
    node: models.ASTNode, span: list[int]
) -> models.StatementNode:
    """Promote a plain node to a StatementNode carrying the span's members.

    Args:
        node: The span's first node.
        span: The span's member node ids.

    Returns:
        The promoted statement node.
    """
    return models.StatementNode(
        content=node.content,
        id=node.id,
        segment_index=node.segment_index,
        statement_of=span,
    )


async def type_roles(
    spans: list[list[int]],
    nodes_by_id: dict[int, models.ASTNode],
    module: RoleTyper | None = None,
) -> tuple[list[int], list[int]]:
    """Diagnose each group's composition and produce statement/procedure ids.

    Every group produces a StatementNode (real or placeholder). Groups with a
    procedure portion get a Procedure attached (real or placeholder).

    Args:
        spans: The untyped spans, each a list of member node ids.
        nodes_by_id: The full node stream keyed by stable id, mutated in place
            as spans are promoted to statement nodes.
        module: The role-typing module. Created fresh if None.

    Returns:
        The ``(statement_ids, procedure_ids)`` pair, both lists of first-node
        ids.
    """
    if not spans:
        logger.info('role typer: no spans')
        return [], []
    module = module or RoleTyper()

    roles = await asyncio.gather(
        *(module.acall(contents_of(span, nodes_by_id)) for span in spans)
    )
    statement_ids: list[int] = []
    procedure_ids: list[int] = []
    for span, role in zip(spans, roles, strict=True):
        first = span[0]
        if first not in nodes_by_id:
            continue
        statement = _mark_statement(nodes_by_id[first], span)
        nodes_by_id[first] = statement
        statement_ids.append(first)
        if role == PROCEDURE_ROLE:
            statement.procedures.append(models.Procedure(index=0))
            procedure_ids.append(first)

    logger.info(
        'role typer: %d span(s) -> %d statement(s), %d procedure(s)',
        len(spans),
        len(statement_ids),
        len(procedure_ids),
    )
    return statement_ids, procedure_ids


# --- LangGraph node: split the untyped spans into blocks and derivations ---


class RoleTyperNode:
    """Diagnoses each group and creates its StatementNode and Procedure.

    Produces ``statement_ids`` (every group) and ``procedure_ids`` (the subset
    with procedures) for the extractors.

    Args:
        module: The role-typing module. Created fresh if None.
    """

    def __init__(self, module: RoleTyper | None = None) -> None:
        self.module = module or RoleTyper()

    async def run(self, state: state.State) -> dict:
        """Diagnose groups and produce statement/procedure ids.

        Args:
            state: The pipeline state, holding the node stream and spans.

        Returns:
            The `statement_ids` and `procedure_ids` channels.
        """
        nodes = state.get('nodes', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        statement_ids, procedure_ids = await type_roles(
            state.get('spans', []), nodes_by_id, self.module
        )
        for position, node in enumerate(nodes):
            if node.id is not None and node.id in nodes_by_id:
                nodes[position] = nodes_by_id[node.id]
        return {
            'statement_ids': statement_ids,
            'procedure_ids': procedure_ids,
        }
