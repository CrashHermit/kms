r"""
Block typer — induces the OPEN ``type`` of each pedagogical block.

One question, asked once per block: what KIND of block is this? ``definition`` /
``theorem`` / ``example`` / ``exercise`` / ``law`` / ``mechanism`` / … The vocabulary is
OPEN and induced from the book itself, never a closed enum and never a Neo4j label — open
label sets explode, so **kind is a label, type is a property** (``docs/SCHEMA.md``). That is
what lets a physics or biology book type itself with no profile.

WHY THIS IS ITS OWN STAGE. ``type`` used to be one of four output fields on the statement
extractor's single pass, sharing a call with ``label`` / ``number`` / ``title``. Those are
different kinds of work: label/number/title are *transcription* — copy what leads the block
— while ``type`` is a *judgement* about what the block is for. Fusing them made the
judgement the junior partner in a call mostly about copying, and it typed a problem set's
items by their subject matter (an exercise whose body is a mathematical claim came back
``theorem``). On its own, with one thing to decide, the pass is markedly more accurate.

It runs AFTER ``role_typer``, so every block it sees is already known to be a block rather
than a derivation — it never has to consider "proof" or "solution" as a type. A derivation's
kind is deliberately not stored at all: it is derivable from the owning block's type
(``docs/SCHEMA.md``, principle 5).

Entry point ``type_blocks(entities, nodes_by_id)`` (async): writes ``type`` onto each passed
entity. Persistence-agnostic.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)


class MemberNode(BaseModel):
    """One member node as the typer sees it: a local position and its content."""

    position: int
    type: str
    content: str | None = None


class Induce(dspy.Signature):
    r"""
    Read one pedagogical BLOCK from a textbook — given as an ordered list of its member
    nodes — and name what KIND of block it is, as a single lowercase word or short phrase.
    This is domain-neutral: the block may come from a math, physics, CS, or biology
    textbook.

    Use the word the book itself uses where there is one: "definition", "theorem",
    "proposition", "lemma", "corollary", "axiom", "example", "exercise", "problem", "law",
    "principle", "rule", "model", "mechanism". Do NOT force it into a fixed list — if a book
    presents a block as a "key concept" or an "investigation", say so. Judge by what the
    block DOES: a block that states something is true is a definition/theorem/law; a block
    that poses a task is an example/exercise/problem.

    TYPE THE BLOCK, NOT ITS SUBJECT MATTER. Judge what the block IS in the book, never what
    its content is about. A block whose body happens to be a mathematical assertion is NOT
    thereby a theorem — an exercise reading "For matrix $A$ to be invertible it is necessary
    and sufficient that $\det(A) \neq 0$" is an EXERCISE (the reader is being asked to do
    something with that sentence), and a quoted line of prose given as an exercise item is an
    EXERCISE, not a "quote". Likewise a bare expression with no instruction of its own —
    "$P \vee (Q \Rightarrow R)$", "$y = \sqrt{x}$" — is an EXERCISE: it is an item in a
    problem set whose shared directive sits in a lead-in you cannot see. Do not call it an
    "example".

    EXAMPLE vs EXERCISE — the test is whether the WORKING IS SHOWN, not the phrasing:
    - "example" — the book works it out FOR the reader (it is labelled "Example ...", and/or
      a solution or derivation accompanies it).
    - "exercise" / "problem" — it is posed FOR THE READER to do, with no working shown. A
      block led by a BARE NUMBER with no type word ("1.", "12.", "2.1.12") is an exercise,
      not an example.

    This block is already known NOT to be a proof or a solution, so never answer "proof",
    "solution", or "derivation".
    """

    nodes: list[MemberNode] = dspy.InputField(
        description="The block's member nodes, in order."
    )
    type: str = dspy.OutputField(
        description='What kind of block this is, lowercase (definition / theorem / example / '
        'exercise / law / …). A single word or short phrase.'
    )


class Module(dspy.Module):
    """Induces the open ``type`` for one pedagogical block."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.induce = dspy.ChainOfThought(Induce)
        self.set_lm(language_model or llm.text_lm())

    async def block_type(self, members: list[models.ASTNode]) -> str | None:
        """Returns the induced type for one block, or None if the model gave nothing."""
        nodes = [
            MemberNode(
                position=k,
                type=(member.type.value if member.type else ''),
                content=member.content,
            )
            for k, member in enumerate(members)
        ]
        result = await self.induce.acall(nodes=nodes)
        induced = normalize_type(result.type) or None
        logger.debug(
            'type: %d node(s) -> %r | from %r',
            len(members),
            induced,
            logs.elide(members[0].content if members else ''),
        )
        return induced


def normalize_type(raw: str | None) -> str:
    """Lowercase and whitespace-collapse an induced type. Open vocabulary, so this only
    normalises the spelling — it never validates against a list."""
    return ' '.join((raw or '').split()).lower()


async def type_blocks(
    entities: list[models.Entity],
    nodes_by_id: dict[int, models.ASTNode],
    module: Module | None = None,
) -> list[models.Entity]:
    """Induce and write the open ``type`` on every block, in place.

    Each block is typed independently, so the calls run concurrently.

    Args:
        entities: The block overlay from the role typer (members only).
        nodes_by_id: The full node stream keyed by stable id.
        module: The typer module. Created fresh if None.

    Returns:
        The same entities, with ``type`` filled in.
    """
    module = module or Module()
    if not entities:
        logger.info('block typer: no blocks')
        return entities

    types = await asyncio.gather(
        *(
            module.block_type(members_of(entity, nodes_by_id))
            for entity in entities
        )
    )
    for entity, induced in zip(entities, types, strict=True):
        entity.type = induced

    logger.info(
        'block typer: %d block(s) typed | %s',
        len(entities),
        logs.counts([entity.type for entity in entities]),
    )
    return entities


def members_of(
    entity: models.Entity, nodes_by_id: dict[int, models.ASTNode]
) -> list[models.ASTNode]:
    """The entity's member nodes, in member order, skipping any id not in the stream."""
    return [nodes_by_id[i] for i in entity.members if i in nodes_by_id]


# --- LangGraph node: induce each block's open type ---


class BlockTyperNode:
    """Induces the open ``type`` on each block in the overlay, in place.

    Runs after the role typer (so every entity is known to be a block, not a derivation) and
    before the statement extractor, which fills the remaining attributes."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Induces each block's open type."""
        nodes_by_id = {
            node.id: node
            for node in state.get('nodes', [])
            if node.id is not None
        }
        entities = state.get('entities', [])
        await type_blocks(entities, nodes_by_id, self.module)
        return {'entities': entities}
