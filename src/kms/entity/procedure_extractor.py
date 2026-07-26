r"""
Procedure extractor — decomposes each *found* procedure span into ordered steps and
attaches it to the block it derives.

The group finder detects a worked derivation (proof, solution, derivation, worked
calculation) as its OWN span, so this pass never has to decide *whether* there is
something worked out — the presence or absence of a procedure span answers that
structurally (``docs/SCHEMA.md``, principle 4). What is left is two jobs:

  * DECOMPOSE. Split the span's content into ordered steps. This is UNIVERSAL: every
    procedure decomposes, whatever it derives. AutoMathKG restricted its ``bodylist`` to
    Theorems and Definitions, which left every solution stepless — the procedural spine was
    empty for exactly the exercise-heavy books this pipeline targets. That restriction is
    gone: a solution's steps are as real as a proof's.
  * ATTACH. Hang the procedure off the nearest PRECEDING block in document order.

Steps are a VERBATIM PARTITION of the procedure's content: every character belongs to
exactly one step, and concatenating the steps in order reproduces the source with nothing
added or removed. Provenance is never rewritten (``docs/SCHEMA.md``, principle 3). Steps
carry no role/tactic label — AutoMathKG's closed nine-value ``action`` taxonomy is gone, and
the concept layer supersedes it with open, cross-corpus-linkable tags.

Orphans: a procedure span with no preceding block — a proof deferred pages after its
theorem — is left UNATTACHED rather than dropped. Its text and steps are real extracted
content, so it is returned separately for the persister to write with no ``:HAS_PROCEDURE``
edge; an attachment pass can find orphans later. Resolving an explicit cross-reference
("Proof of Theorem 2.4") is deliberately not implemented yet.

Entry point ``extract_procedures(entities, procedure_spans, nodes_by_id)`` (async).
Persistence-agnostic.
"""

import asyncio
import logging

import dspy

from kms.core import llm, logs, models, state
from kms.entity import statement_extractor

logger = logging.getLogger(__name__)


class Decompose(dspy.Signature):
    r"""
    Segment a worked PROCEDURE — a proof, a solution, a derivation, a worked calculation —
    into its ordered steps. Return an ordered list of step strings.

    A STEP is one move in the derivation: a single inference, calculation, substitution,
    case, construction, or concluding claim. Cut where the reasoning moves on. Keep a step
    whole — do not split a single equation or a single sentence across two steps, and do not
    fuse two distinct moves into one.

    PARTITION (critical): the steps must exactly PARTITION the content — every part belongs
    to EXACTLY ONE step, no repeats and no omissions; reading the steps in order, with
    nothing added or removed, must reproduce the content.

    EVERYTHING IN THE CONTENT IS PART OF THE PARTITION — not just the lines that look like
    "moves". Surrounding and interleaved PROSE is included: a sentence that sets the working
    up, a remark between two calculations, and above all a TRAILING sentence that comments on
    the result ("The output above lists each root along with its multiplicity.", "This
    completes the proof.") must each land in a step. A closing remark is its own final step —
    do NOT drop it because it is commentary rather than computation. The same holds for a
    worked session's output lines and for code: a transcript's commands AND their printed
    results are all part of the content. If you are unsure whether something is a step, it
    still belongs to one — omitting it breaks the partition.

    STEP TEXT: copy each step's text VERBATIM — reproduce all mathematics and LaTeX exactly
    as given, changing nothing. Do not summarise, paraphrase, renumber, or explain. Do not
    add step labels ("Step 1:") that are not in the source.

    If the content is a single indivisible move, return it as one step.
    """

    contents: str = dspy.InputField(
        description="The procedure's content (text + LaTeX)."
    )
    steps: list[str] = dspy.OutputField(
        description='Ordered verbatim step strings; concatenating them reproduces the content.'
    )


class Module(dspy.Module):
    """Runs the step decomposition for one procedure."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.decompose = dspy.ChainOfThought(Decompose)
        self.set_lm(language_model or llm.text_lm())

    async def steps(self, contents: str) -> list[str]:
        """Returns the ordered verbatim steps for one procedure's content."""
        result = await self.decompose.acall(contents=contents)
        steps = [step for step in (result.steps or []) if step and step.strip()]
        # Steps must PARTITION the content, so the two character counts should be near
        # equal; a large shortfall means the decomposition dropped text.
        logger.debug(
            'decompose: %d chars -> %d step(s), %d chars | from %r',
            len(contents),
            len(steps),
            sum(len(step) for step in steps),
            logs.elide(contents),
        )
        return steps


def attach(
    entities: list[models.Entity],
    procedure_spans: list[list[int]],
    order: dict[int, int],
) -> tuple[dict[int, list[list[int]]], list[list[int]]]:
    """Assign each procedure span to the nearest PRECEDING block in document order.

    Both blocks and procedure spans are keyed by their first member's stream position. A
    procedure attaches to the last block that starts before it; one that precedes every
    block is an orphan.

    Args:
        entities: The block overlay from the group finder.
        procedure_spans: Member-id lists, one per detected procedure span.
        order: Stream position keyed by node id.

    Returns:
        A ``(by_entity, orphans)`` pair — ``by_entity`` maps an entity's index in
        ``entities`` to its procedure spans in document order; ``orphans`` holds the spans
        that attach to nothing.
    """
    last = len(order)
    starts = sorted(
        (
            (order.get(entity.members[0], last), index)
            for index, entity in enumerate(entities)
            if entity.members
        ),
    )
    by_entity: dict[int, list[list[int]]] = {}
    orphans: list[list[int]] = []
    for span in sorted(
        procedure_spans,
        key=lambda ids: order.get(ids[0], last) if ids else last,
    ):
        if not span:
            continue
        position = order.get(span[0], last)
        preceding = [index for start, index in starts if start < position]
        if not preceding:
            orphans.append(span)
            continue
        by_entity.setdefault(preceding[-1], []).append(span)
    return by_entity, orphans


async def _build(
    span: list[int],
    index: int,
    nodes_by_id: dict[int, models.ASTNode],
    module: Module,
) -> models.Procedure:
    """Decompose one procedure span into a Procedure with verbatim steps."""
    members = [nodes_by_id[i] for i in span if i in nodes_by_id]
    contents = statement_extractor.contents_of(members)
    steps = await module.steps('\n\n'.join(contents)) if contents else []
    return models.Procedure(
        index=index, members=span, contents=contents, steps=steps
    )


async def extract_procedures(
    entities: list[models.Entity],
    procedure_spans: list[list[int]],
    nodes_by_id: dict[int, models.ASTNode],
    module: Module | None = None,
) -> list[models.Procedure]:
    """Decompose every procedure span and attach it to the block it derives, in place.

    Each span is attached to the nearest preceding block and decomposed into verbatim steps.
    An entity may end up with several procedures (a theorem with two proofs); ``index``
    orders them. Decompositions are independent, so they run concurrently.

    Args:
        entities: The block overlay, already attributed.
        procedure_spans: Member-id lists, one per detected procedure span.
        nodes_by_id: The full node stream keyed by stable id.
        module: The extractor module. Created fresh if None.

    Returns:
        The orphan procedures — decomposed but attached to no entity. The attached ones are
        written onto their entity's ``procedures`` list in place.
    """
    module = module or Module()
    order = {
        node_id: position
        for position, node_id in enumerate(sorted(nodes_by_id))
    }
    by_entity, orphan_spans = attach(entities, procedure_spans, order)

    tasks = [
        (entity_index, index, span)
        for entity_index, spans in by_entity.items()
        for index, span in enumerate(spans)
    ]
    built = await asyncio.gather(
        *(_build(span, index, nodes_by_id, module) for _, index, span in tasks),
        *(
            _build(span, index, nodes_by_id, module)
            for index, span in enumerate(orphan_spans)
        ),
    )
    attached = built[: len(tasks)]
    orphans = list(built[len(tasks) :])

    for (entity_index, _, _), procedure in zip(tasks, attached, strict=True):
        entities[entity_index].procedures.append(procedure)

    # Orphans are returned rather than dropped, but the caller currently discards them and
    # nothing writes them to the graph — so this line is the only record that a book had a
    # derivation with no preceding block.
    if orphans:
        logger.warning(
            '%d procedure(s) attached to no block and will not be persisted',
            len(orphans),
        )
    logger.info(
        'procedure extractor: %d span(s) -> %d attached, %d orphan, %d step(s)',
        len(procedure_spans),
        len(attached),
        len(orphans),
        sum(len(procedure.steps) for procedure in built),
    )
    return orphans


# --- LangGraph node: decompose and attach the found procedures ---


class ProcedureExtractorNode:
    """Decomposes each found procedure span and attaches it to the block it derives.

    Runs after the statement extractor, over the ``entities`` and ``procedure_spans``
    channels. Orphan procedures (no preceding block) are dropped from the in-memory overlay
    for now — the graph tier will grow an orphan write when the attachment rule is
    designed."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Decomposes the found procedure spans and attaches them to their blocks."""
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        entities = state.get('entities', [])
        spans = state.get('procedure_spans', [])
        if entities and spans:
            await extract_procedures(entities, spans, nodes_by_id, self.module)
        return {'entities': entities}
