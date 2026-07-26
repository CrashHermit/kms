r"""
Instruction distributor — copies a grouped-exercise lead-in's shared directive onto the
blocks it governs, by a growing-window walk (no number/range parsing).

A run of exercises often states its imperative ONCE, in a lead-in that governs the whole run.
That lead-in is not a member of any individual exercise, so the per-entity statement
extractor can't see it — `instruction` is inherently a cross-entity, positional attribute.
The instruction finder has already TAGGED each lead-in node `role="instruction"`; this pass
reads those tags and distributes the directive.

It is what makes an atomic exercise mean anything: "3. y = x^2 + 1" retrieved on its own is
noise, while the same block carrying "graph the following relations" is a usable item. That
is why the directive is copied onto every governed block rather than left on the lead-in.

The extent is judged by the LLM, not by numbers. A lead-in may name a range ("In Exercises
1.23-1.25, …") or may not ("Answer the following.", "Prove each of the following."), so a range
parser can't decide who is governed — the model reads the following problems and decides where
the governed run ends. It is the SAME growing look-ahead used by the finders and the splitter:

  * Anchor on a tagged lead-in node. Its candidate problems are the ones that follow it, up to
    the next lead-in (a new lead-in starts a new governance).
  * Take a look-ahead window of those following problems (whole problems up to a token budget)
    and ask the LLM which of them the lead-in governs, plus the shared instruction to apply.
  * If the governed run reaches the window's edge it may continue, so GROW the window (double
    the budget) and re-read; if a non-governed problem is seen to follow the run (bounded), or
    the candidates are exhausted, BANK — stamp `entity.instruction` on the governed problems.

It runs AFTER the statement extractor (it reads each block's `contents`/`number` to judge
governance). Entry point ``distribute_instructions(nodes, blocks, module)`` mutates the
blocks in place; ``InstructionDistributorNode`` wires it onto the ``entities`` channel.
"""

import logging

import dspy
from pydantic import BaseModel

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)

# Same growing look-ahead shape as the finders/splitter (~4 chars/token).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowProblem(BaseModel):
    """One following problem as the LLM sees it: a local position, its number, its statement."""

    position: int
    number: str | None = None
    text: str | None = None


class GovernExtent(dspy.Signature):
    r"""
    Given an exercise LEAD-IN and the problems that FOLLOW it in document order, decide which of
    those problems the lead-in's shared instruction governs, and give the instruction to apply.

    A lead-in states one imperative for a run of exercises. Some name a range ("In Exercises
    1.23-1.25, find the eigenvalues of each matrix."), some do not ("Answer the following.",
    "Prove each of the following statements."). Judge governance by MEANING, not by numbers: the
    governed problems are a run that STARTS at the first following problem and continues while
    the lead-in's instruction still sensibly applies to them, and STOPS when it no longer does —
    a problem that is clearly a different task, or the start of a different group.

    Return:
      * instruction — the shared imperative to apply to the governed problems, copied as written
        but WITHOUT any "In Exercises X-Y," framing ("find the eigenvalues of each matrix",
        "answer the following"). EMPTY string if the lead-in actually governs nothing here.
      * governed_positions — the `position` values of the governed problems (a run from the
        first). EMPTY list if none are governed.
    """

    lead_in: str = dspy.InputField(description="The lead-in node's text.")
    following_problems: list[WindowProblem] = dspy.InputField(
        description='The problems that follow the lead-in, in order, each with a local position.'
    )
    instruction: str = dspy.OutputField(
        description='The shared imperative to apply, without range framing, or empty string.'
    )
    governed_positions: list[int] = dspy.OutputField(
        description='Positions of the governed problems, a run from the first; empty if none.'
    )


class Module(dspy.Module):
    """Determines which following problems a lead-in's shared instruction governs."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(GovernExtent)
        self.set_lm(language_model or llm.text_lm())

    async def govern(
        self, lead_in: str, following: list[WindowProblem]
    ) -> tuple[str, list[int]]:
        """Returns the shared instruction and the positions it governs."""
        result = await self.judge.acall(
            lead_in=lead_in, following_problems=following
        )
        instruction, positions = (
            (result.instruction or '').strip(),
            list(result.governed_positions or []),
        )
        # Extent is the judgement worth auditing: the run a lead-in governs is decided
        # here, never by parsing its numbers.
        logger.debug(
            'govern: %d candidate(s) -> position(s) %s | instruction %r',
            len(following),
            positions or 'none',
            logs.elide(instruction),
        )
        return instruction, positions


def _estimate_tokens(block: models.Entity) -> int:
    return len(_block_text(block)) // 4 + 1


def _block_text(block: models.Entity) -> str:
    """The block's statement as the LLM should see it — its attributed contents, or its
    number as a last resort."""
    body = ' '.join(text for text in (block.contents or []) if text)
    return body or (block.number or '')


def _window(
    candidates: list[models.Entity], budget: int
) -> list[models.Entity]:
    """Whole following blocks up to the soft token budget, always at least one."""
    window, accumulated = [], 0
    for block in candidates:
        token_count = _estimate_tokens(block)
        if window and accumulated + token_count > budget:
            break
        window.append(block)
        accumulated += token_count
    return window


async def _govern_one(
    node: models.ASTNode, candidates: list[models.Entity], module: Module
) -> None:
    """Growing-window walk for one lead-in: find the governed run among the blocks that follow
    it and stamp the instruction on them. Grows the window while the run reaches its edge."""
    if not candidates:
        return
    size = LOOKAHEAD_BUDGET
    while True:
        window = _window(candidates, size)
        last_local = len(window) - 1
        exhausted = len(window) == len(candidates)

        instruction, positions = await module.govern(
            node.content or '',
            [
                WindowProblem(
                    position=k,
                    number=window[k].number,
                    text=_block_text(window[k]),
                )
                for k in range(len(window))
            ],
        )
        governed = sorted(
            {min(max(position, 0), last_local) for position in positions}
        )

        if not governed:
            return  # this lead-in governs nothing here
        run_end = governed[-1]

        if exhausted or size >= MAX_LOOKAHEAD_BUDGET or run_end < last_local:
            # Bounded (a non-governed problem follows the run), or nothing left to gather: bank.
            if instruction:
                for k in governed:
                    window[k].instruction = instruction
            return
        size *= 2  # the run reaches the window edge and may continue — grow and re-read


async def distribute_instructions(
    nodes: list[models.ASTNode],
    blocks: list[models.Entity],
    module: Module | None = None,
) -> list[models.Entity]:
    """Stamp each governed block's `instruction` from the tagged lead-in nodes, in place.

    The governed run is judged per lead-in with a growing-window LLM walk, never by parsing
    numbers — a lead-in may name a range or none at all.

    Args:
        nodes: The flat node stream, carrying the lead-in `role` tags.
        blocks: The attributed block overlay, mutated in place.
        module: The governance module. Created fresh if None.

    Returns:
        The same blocks, with `instruction` stamped on the governed ones.
    """
    lead_ins = [node for node in nodes if node.role == 'instruction']
    if not lead_ins or not blocks:
        logger.info(
            'instruction distributor: no-op (%d lead-in(s), %d block(s))',
            len(lead_ins),
            len(blocks),
        )
        return blocks
    module = module or Module()
    order = {node.id: i for i, node in enumerate(nodes)}
    last = len(order)

    def position_of(entity: models.Entity) -> int:
        """The entity's first member's position in the stream; `last` if it has none."""
        return order.get(entity.members[0], last) if entity.members else last

    ordered = sorted(blocks, key=position_of)
    lead_positions = sorted(order.get(node.id, -1) for node in lead_ins)

    for node in lead_ins:
        here = order.get(node.id, -1)
        # A new lead-in starts a new governance, so a lead-in's candidates end at the next one.
        next_lead_in = min(
            (position for position in lead_positions if position > here),
            default=last,
        )
        candidates = [
            block
            for block in ordered
            if here < position_of(block) < next_lead_in
        ]
        await _govern_one(node, candidates, module)
    logger.info(
        'instruction distributor: %d lead-in(s) -> %d of %d block(s) stamped',
        len(lead_ins),
        sum(1 for block in blocks if block.instruction),
        len(blocks),
    )
    return blocks


# --- LangGraph node: distribute instructions over the attributed blocks ---


class InstructionDistributorNode:
    """Stamps `instruction` onto the governed blocks, reading the instruction finder's
    `role="instruction"` lead-in tags. Runs after the statement extractor (it reads each
    block's contents/number) and writes the enriched entities back to the ``entities``
    channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Distributes each lead-in's shared instruction onto the governed blocks."""
        entities = state.get('entities', [])
        await distribute_instructions(
            state.get('nodes', []), entities, module=self.module
        )
        return {'entities': entities}
