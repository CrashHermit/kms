r"""
Statement extractor — the attribute pass over a *typed* pedagogical block.

One universal pass replacing the three per-type attributors (definition / theorem /
problem). It reads a block's member nodes and fills its remaining attributes:

    label · number · title · contents

All four are TRANSCRIPTION rather than judgement — copy the label that leads the block, the
reference number inside it, a short descriptive title, and assemble the contents. The block's
kind is NOT decided here: ``role_typer`` has already established it is a block rather than a
derivation, and ``block_typer`` has induced its open ``type``. Those are separate stages
precisely because judging a block's kind and copying its header are different jobs, and
fusing them made the judgement the junior partner (it typed problem-set items by their
subject matter).

What this pass no longer does, and why:
  * NO ``type``. Induced upstream by ``block_typer``, on its own, one question per block.
  * NO ``field``. AutoMathKG's closed 7-value mathematical-field taxonomy is gone; the
    concept layer subsumes it with open, multi-granularity induced concepts.
  * NO ``bodylist``. The statement's role-labelled segmentation (premise / assumption /
    conclusion) was written to Neo4j as a JSON string and never read back by anything.
  * NO ``proof_start`` / ``solution_start``, and no member splitting. The group finder cuts
    the derivation into its own span, so the statement/procedure boundary is a structural
    detection rather than a semantic call made here. This pass reads a span and fills
    attributes; it never restructures the entity.

Entry point ``extract_statement(entity, nodes_by_id)`` (async): writes the attributes onto
the passed entity and returns it. Persistence-agnostic.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)


class MemberNode(BaseModel):
    """One member node as the extractor sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class Identify(dspy.Signature):
    r"""
    Read a single pedagogical BLOCK from a textbook — given as an ordered list of its member
    nodes — and transcribe its header information. This is domain-neutral: the block may come
    from a math, physics, CS, or biology textbook.

    This is TRANSCRIPTION, not judgement: copy what leads the block. What KIND of block it is
    has already been decided by an earlier pass — do not reason about it here.

      * label — the block's own label as it appears at the very START of the block
        ("Example 4.1", "Theorem 2.5.8", "Definition 3.1", "Exercise 12"), INCLUDING a bare
        leading reference number carrying no word ("925.", "3.14", "2.1.12"). Read only what
        LEADS the first member node; empty string if it carries no label.
      * number — just the reference number in that LEADING label ("4.1", "12", "3", "925",
        "2.1.12"). This is the block's OWN number at its start — NEVER a number that appears
        later inside the text as a cross-reference to another result. In "2.1.12 Prove
        Proposition 2.1.13." the number is 2.1.12, not 2.1.13; in "3.15 ... use Theorem 3.7"
        it is 3.15, not 3.7. Empty if there is none.
      * title — a short noun phrase naming what the block is about ("Positive Definiteness
        of a Matrix", "Derivative of a Polynomial", "Second Law of Thermodynamics"). Not the
        word "Example", "Exercise", "Theorem" or "Definition" on its own.
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


class Identity(BaseModel):
    """The extraction pass's result for one block."""

    label: str | None = None
    number: str | None = None
    title: str | None = None


class StatementExtractor(dspy.Module):
    """Runs the attribute transcription pass for one pedagogical block."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.identify = dspy.Predict(Identify)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, members: list[models.ASTNode]) -> Identity:
        """Returns the label, number and title for one block."""
        nodes = [
            MemberNode(
                position=k,
                type=(member.type.value if member.type else ''),
                content=member.content,
            )
            for k, member in enumerate(members)
        ]
        result = await self.identify.acall(nodes=nodes)
        identity = Identity(
            label=(result.label or None),
            number=(result.number or None),
            title=(result.title or None),
        )
        logger.debug(
            'identify: %d node(s) -> label=%r number=%r title=%r',
            len(members),
            identity.label,
            identity.number,
            logs.elide(identity.title, 40),
        )
        return identity


def members_of(
    entity: models.Entity, nodes_by_id: dict[int, models.ASTNode]
) -> list[models.ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


def contents_of(
    members: list[models.ASTNode], label: str | None = None
) -> list[str]:
    """The content members as a list of sequence strings, with `label` peeled off the front.

    A standalone label node ("Example 4.1") strips to empty and is dropped; a fused label
    ("Example 4.1. Find ...") leaves its statement, which is kept; a content-bearing node is
    never dropped wholesale. Passing ``label=None`` peels nothing."""
    texts = [
        member.content
        for member in members
        if member.content and member.content.strip()
    ]
    if (
        texts and label
    ):  # peel the label off the first content piece; drop it if that empties it
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
    stripped = label.strip().rstrip('.')
    if stripped and body[: len(stripped)].lower() == stripped.lower():
        return body[len(stripped) :].lstrip(' .:\t\n')
    return text


async def extract_statement(
    entity: models.Entity,
    nodes_by_id: dict[int, models.ASTNode],
    module: StatementExtractor | None = None,
) -> models.Entity:
    """Fill in the self-contained attributes on one pedagogical block, in place.

    A single LLM call gives label/number/title; ``contents`` is then assembled
    deterministically from the members with the label peeled off. The block's extent is
    exactly what the finder detected — this pass never splits or reorders members.

    Args:
        entity: The block, already role-typed and type-induced.
        nodes_by_id: The full node stream keyed by stable id.
        module: The extractor module. Created fresh if None.

    Returns:
        The same entity, with label, number, title and contents filled in.
    """
    module = module or StatementExtractor()
    members = members_of(entity, nodes_by_id)
    identity = await module.acall(members)

    entity.label = identity.label
    entity.number = identity.number
    entity.title = identity.title
    entity.contents = contents_of(members, identity.label)
    return entity


# --- LangGraph node: fill in each found block's attributes ---


class StatementExtractorNode:
    """Fills in each block's remaining attributes, in place.

    Runs after the block typer, over the ``entities`` channel. The per-entity
    extractions are independent, so they run concurrently; the enriched entities (mutated in
    place) are written back to the same channel."""

    def __init__(self, module: StatementExtractor | None = None) -> None:
        self.module = module or StatementExtractor()

    async def run(self, state: state.State) -> dict:
        """Fills in each found block's self-contained attributes, in place."""
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        entities = state.get('entities', [])
        if entities:
            await asyncio.gather(
                *(
                    extract_statement(entity, nodes_by_id, self.module)
                    for entity in entities
                )
            )
        logger.info(
            'statement extractor: %d block(s) attributed', len(entities)
        )
        return {'entities': entities}
