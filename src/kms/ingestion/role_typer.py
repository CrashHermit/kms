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

The distinction is a UNION of two roles — exactly three answers: a block can
contain a ``statement`` (a block), a ``procedure`` (a derivation), or BOTH. A
block that both poses something and works it out gets both roles; the
statement/procedure partitioners then find the line between the portions.

Absence stays structural (``docs/SCHEMA.md``, principle 4): this pass never asks
"is there something to work out?" — it only labels the spans it is given. A unit
the book never works out simply produces no procedure-role span, because the
finder never cut one.

THE OVERLAY IS NOT PART OF THE NODE STREAM. A span's ``models.Statement`` is
built FROM its first node but never put back in that node's place: it is
returned on its own ``statements`` channel, and ``nodes`` comes out of this
stage exactly as it went in. The two say different things — a node is one
verbatim block of the page (the provenance tier), a statement is the whole
group the span covers — and a statement standing in the stream would make every
stage that walks ``nodes`` read the group's text twice, once inside the
statement and once from the members that follow it. The assembler walking that
stream is where it showed: every member after a group's first was emitted twice
into the assembled output. ``Statement`` is no longer an ``ASTNode`` at all, so the
stream can no longer hold one.

Entry point ``type_roles(spans, nodes_by_id)`` (async): returns the statement
and procedure hub overlays the rest of the pipeline consumes. Persistence-
agnostic.
"""

import asyncio
import logging

import dspy

from kms.core import llm, logs, models, recorder, state

logger = logging.getLogger(__name__)

# The two roles. A union: a block may contain a statement, a procedure, or
# both — the role typer's answer is the subset present.
STATEMENT_ROLE = 'statement'
PROCEDURE_ROLE = 'procedure'
SPAN_ROLES = [STATEMENT_ROLE, PROCEDURE_ROLE]


class Classify(dspy.Signature):
    r"""
    Decide what a pedagogical block contains. Answer with the subset of two
    roles present in the block — "statement", "procedure", or both. This is
    domain-neutral: the text may come from a math, physics, CS, or biology
    textbook.

    "statement" — the block STATES something. It says that something is so, or
    asks for something to be done: a claim, a definition, a problem posed to
    the reader. If the block opens with its own label naming it as a unit of
    the book — "Definition 2.5.1", "Theorem 3.4", "Example 6.7", "Exercise
    12", or a bare leading number ("12.", "2.1.12") — it states something,
    whatever follows that label.

    "procedure" — the block WORKS something out: a proof, a solution, a
    derivation, a worked calculation that resolves what a block before it
    stated. Signs of working: substituting, integrating, factoring, splitting
    into cases, applying a named result, computing, concluding ("hence",
    "therefore", "so we get", "this completes the proof").

    WORKING IS NOT ONLY ALGEBRA. Text that RESOLVES a statement is a procedure
    even when it manipulates no symbols at all: exhibiting an answer ("Note
    that $y = 0$ is a solution. But another solution is the function ..."),
    analysing the posed case or figure, verifying or justifying ("$G_4$ is NOT
    a subgraph, because ..."). Ask "does this text work out what came before
    it?" — not "does it contain equations?".

    A COMPUTATION SESSION IS A PROCEDURE. Unlabelled transcript lines and their
    printed output — "sage: f = x^15 + 1", "sage: f.roots()", "[(12, 1), (10,
    1), (4, 1)]", a shell or REPL session, a table of computed values — are
    the working of a block above them, so they are "procedure".

    A derivation never carries a block label of its own — it either opens with
    a derivation marker ("Proof.", "Solution.") or is unlabelled text
    continuing from the block before it. For unlabelled text, never answer
    "statement" only because a marker word is missing.

    Judge the block in front of you on its own terms. If it only states
    something, include only "statement". If it only works something out,
    include only "procedure". If it both states something and then works it
    out, include BOTH.
    """

    contents: str = dspy.InputField(
        description="The span's text (markdown + LaTeX), in document order."
    )
    roles: list[str] = dspy.OutputField(
        description='Exactly the subset of roles the block contains: "statement" (states something) or "procedure" (works something out). Both when it states something and then works it out.'
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

    async def aforward(self, contents: str) -> list[str]:
        """Classify one span.

        Args:
            contents: The span's text, in document order.

        Returns:
            The span's roles — a subset of ``{statement, procedure}``,
            falling back to ``['statement']`` for an unusable answer.
        """
        result = await self.classify.acall(contents=contents)
        recorder.record_example('role_typer', {'contents': contents}, result)
        roles = sorted(
            {
                ' '.join(role.split()).lower()
                for role in (result.roles or [])
                if ' '.join(role.split()).lower() in SPAN_ROLES
            }
        )
        # `statement` is the fallback: it is the far more common role, and a
        # block wrongly demoted to a derivation would lose its statement hub
        # entirely.
        if not roles:
            roles = [STATEMENT_ROLE]
        logger.debug(
            'roles: %s%s | from %r',
            roles,
            '' if roles else f' (fallback, model said {result.roles!r})',
            logs.elide(contents),
        )
        return roles

    def forward(self, contents: str) -> list[str]:
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


def _mark_statement(span: list[int]) -> models.Statement:
    """Build the span's Statement hub.

    The hub's identity is the WHOLE span — the block's member node ids, frozen
    at creation (see ``graph.statements.statement_uuid``) — and its members
    start as the whole block until the statement partitioner narrows them to
    the statement portion.

    Args:
        span: The span's member node ids, in document order.

    Returns:
        The span's statement hub.
    """
    return models.Statement(
        block=list(span),
        members=list(span),
    )


def _mark_procedure(span: list[int]) -> models.Procedure:
    """Build the span's Procedure hub.

    The hub's identity is the WHOLE span — the block's member node ids, frozen
    at creation (see ``graph.procedures.procedure_uuid``) — and its members
    start as the whole span until the procedure partitioner narrows them to
    the derivation portion.

    Args:
        span: The span's member node ids, in document order.

    Returns:
        The span's procedure hub.
    """
    return models.Procedure(
        block=list(span),
        members=list(span),
    )


async def type_roles(
    spans: list[list[int]],
    nodes_by_id: dict[int, models.ASTNode],
    module: RoleTyper | None = None,
) -> tuple[list[models.Statement], list[models.Procedure]]:
    """Diagnose each group's composition and build the hub overlays.

    Every PCF collection produces a Statement hub when it contains a
    statement role and a Procedure hub when it contains a procedure role —
    the two are independent, even when they share a block. Nothing here
    writes to the node stream — see the module docstring.

    Args:
        spans: The untyped spans, each a list of member node ids.
        nodes_by_id: The full node stream keyed by stable id. Read-only.
        module: The role-typing module. Created fresh if None.

    Returns:
        The ``(statements, procedures)`` hub overlays, in span order.
    """
    if not spans:
        logger.info('role typer: no spans')
        return [], []
    module = module or RoleTyper()

    roles_by_span = await asyncio.gather(
        *(module.acall(contents_of(span, nodes_by_id)) for span in spans)
    )
    statements: list[models.Statement] = []
    procedures: list[models.Procedure] = []
    for span, roles in zip(spans, roles_by_span, strict=True):
        first_member_id = span[0]
        if first_member_id not in nodes_by_id:
            continue
        # Defensive: a module that returns garbage must not delete the block
        # — `statement` is the far more common role.
        role_set = {role for role in (roles or []) if role in SPAN_ROLES} or {
            STATEMENT_ROLE
        }
        if STATEMENT_ROLE in role_set:
            statements.append(_mark_statement(span))
        if PROCEDURE_ROLE in role_set:
            procedures.append(_mark_procedure(span))

    logger.info(
        'role typer: %d span(s) -> %d statement(s), %d procedure(s)',
        len(spans),
        len(statements),
        len(procedures),
    )
    return statements, procedures


# --- LangGraph node: split the untyped spans into blocks and derivations ---


class RoleTyperNode:
    """Diagnoses each group and creates its Statement and Procedure hubs.

    Produces the ``statements`` and ``procedures`` channels — one hub per
    role present in each group, each holding its members' node ids. The
    ``nodes`` channel is read, never written.

    Args:
        module: The role-typing module. Created fresh if None.
    """

    def __init__(self, module: RoleTyper | None = None) -> None:
        self.module = module or RoleTyper()

    async def run(self, state: state.State) -> dict:
        """Diagnose groups and build the hub overlays.

        Args:
            state: The pipeline state, holding the node stream and spans.

        Returns:
            The `statements` and `procedures` channels. `nodes` is left
            exactly as it was.
        """
        nodes = state.get('nodes', [])
        nodes_by_id = {node.id: node for node in nodes if node.id is not None}
        statements, procedures = await type_roles(
            state.get('spans', []), nodes_by_id, self.module
        )
        return {'statements': statements, 'procedures': procedures}
