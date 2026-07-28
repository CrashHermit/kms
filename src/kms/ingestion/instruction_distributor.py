r"""
Instruction distributor — prepends a grouped-exercise lead-in's shared directive onto the
content of the exercise nodes it governs, then removes the lead-in node from the stream.

The instruction finder has already stamped ``NodeType.INSTRUCTION`` on every lead-in node.
This pass walks the stream, anchors on each INSTRUCTION node, and asks the LLM which of the
following exercise nodes it governs. For each governed exercise, the directive is prepended
to the node's ``content`` so the exercise carries its own context and is usable on its own.
The INSTRUCTION node is then removed from the stream — its job is done.

The extent is judged by the LLM, not by numbers. A lead-in may name a range ("In Exercises
1.23-1.25, …") or may not ("Answer the following."), so a range parser can't decide who is
governed — the model reads the following problems and decides where the governed run ends.
It uses the SAME growing look-ahead pattern as the finders and the splitter:

  * Anchor on an INSTRUCTION node. Its candidates are the non-instruction nodes that follow
    it, up to the next INSTRUCTION (a new lead-in starts a new governance).
  * Take a look-ahead window of those candidates (whole nodes up to a token budget) and ask
    the LLM which of them the lead-in governs, plus the shared instruction to apply.
  * If the governed run reaches the window's edge it may continue, so GROW the window
    (double the budget) and re-read; if a non-governed node follows the run (bounded), or
    the candidates are exhausted, BANK — prepend the instruction to each governed node.
  * After all governance is resolved, delete every INSTRUCTION node from the stream.

Runs BEFORE the node persister — instruction nodes are never written to Neo4j, and the
persisted stream carries exercises with their instruction already prepended to content.
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
    """One following node as the LLM sees it: a local position and its content."""

    position: int
    content: str | None = None


class GovernExtent(dspy.Signature):
    r"""
    Given an exercise LEAD-IN and the exercise nodes that FOLLOW it in document order, decide
    which of those exercises the lead-in's shared instruction governs, and give the
    instruction to apply.

    A lead-in states one imperative for a run of exercises. Some name a range ("In Exercises
    1.23-1.25, find the eigenvalues of each matrix."), some do not ("Answer the following.",
    "Prove each of the following statements."). Judge governance by MEANING, not by numbers:
    the governed exercises are a contiguous run that STARTS at the first exercise after the
    lead-in and continues while the lead-in's instruction still sensibly applies to them,
    and STOPS when it no longer does — an exercise that is clearly a different kind of task,
    or the start of a different group.

    Each following-problem entry contains the exercise's raw text (the exercise number is
    in the text itself). Judge governance from the text content — do not require a separate
    number field.

    Return:
      * instruction — the shared imperative to apply to the governed exercises, copied as
        written but WITHOUT any "In Exercises X-Y," framing ("find the eigenvalues of each
        matrix", "answer the following"). EMPTY string if the lead-in actually governs
        nothing here.
      * governed_positions — the `position` values of the governed exercises (a run from
        the first). EMPTY list if none are governed.
    """

    lead_in: str = dspy.InputField(description="The lead-in node's text.")
    following_problems: list[WindowProblem] = dspy.InputField(
        description='The exercise nodes that follow the lead-in, in order, each with a local position '
        'and its raw text content.'
    )
    instruction: str = dspy.OutputField(
        description='The shared imperative to apply, without range framing, or empty string.'
    )
    governed_positions: list[int] = dspy.OutputField(
        description='Positions of the governed exercises, a run from the first; empty if none.'
    )


class InstructionDistributor(dspy.Module):
    """Determines which following exercises a lead-in's shared instruction governs."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(GovernExtent)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(
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
        logger.debug(
            'govern: %d candidate(s) -> position(s) %s | instruction %r',
            len(following),
            positions or 'none',
            logs.elide(instruction),
        )
        return instruction, positions


def _node_text(node: models.ASTNode) -> str:
    """The node's content as the LLM should see it."""
    return (node.content or '').strip()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _window(
    candidates: list[models.ASTNode], budget: int
) -> list[models.ASTNode]:
    """Whole following nodes up to the soft token budget, always at least one."""
    window, accumulated = [], 0
    for node in candidates:
        token_count = _estimate_tokens(_node_text(node))
        if window and accumulated + token_count > budget:
            break
        window.append(node)
        accumulated += token_count
    return window


async def _govern_one(
    lead_in: models.ASTNode,
    candidates: list[models.ASTNode],
    module: InstructionDistributor,
) -> None:
    """Growing-window walk for one lead-in: find the governed run among the following
    exercise nodes and prepend the instruction to their content."""
    if not candidates:
        return
    size = LOOKAHEAD_BUDGET
    while True:
        window = _window(candidates, size)
        last_local = len(window) - 1
        exhausted = len(window) == len(candidates)

        instruction, positions = await module.aforward(
            lead_in=lead_in.content or '',
            following=[
                WindowProblem(
                    position=k, content=_node_text(window[k])
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
            # Bounded or nothing left to gather: bank.
            if instruction:
                for k in governed:
                    window[k].content = (
                        instruction
                        + '\n\n'
                        + (window[k].content or '')
                    )
            return
        size *= 2  # the run reaches the window edge — grow and re-read


async def distribute_instructions(
    nodes: list[models.ASTNode],
    module: InstructionDistributor | None = None,
) -> list[models.ASTNode]:
    """Walk the node stream, prepend each lead-in's directive onto its governed exercises,
    and remove the lead-in nodes. Returns a new cleaned node list.

    Args:
        nodes: The flat node stream, with INSTRUCTION nodes already tagged.
        module: The governance module. Created fresh if None.

    Returns:
        The cleaned node stream — instruction nodes removed, governed exercises enriched.
    """
    lead_ins = [
        node for node in nodes if isinstance(node, models.InstructionNode)
    ]
    if not lead_ins:
        logger.info(
            'instruction distributor: no-op (0 lead-in(s), %d node(s))',
            len(nodes),
        )
        return nodes

    module = module or InstructionDistributor()

    # Index the stream so we know what follows each lead-in.
    position_of = {node.id: i for i, node in enumerate(nodes) if node.id is not None}
    lead_positions = sorted(
        position_of.get(node.id) for node in lead_ins if node.id is not None
    )
    last = len(nodes)

    for node in lead_ins:
        here = position_of.get(node.id)
        if here is None:
            continue
        # Candidates end at the next lead-in or the end of the stream.
        next_lead = min(
            (p for p in lead_positions if p is not None and p > here),
            default=last,
        )
        candidates = [
            n
            for n in nodes[here + 1 : next_lead]
            if not isinstance(n, models.InstructionNode)
        ]
        await _govern_one(node, candidates, module)

    # Remove instruction nodes from the stream.
    cleaned = [
        node for node in nodes if not isinstance(node, models.InstructionNode)
    ]
    logger.info(
        'instruction distributor: %d lead-in(s) removed, '
        '%d of %d node(s) remain',
        len(lead_ins),
        len(cleaned),
        len(nodes),
    )
    return cleaned


# --- LangGraph node ---


class InstructionDistributorNode:
    """Prepends each lead-in's shared directive onto the governed exercise nodes and
    removes the instruction nodes from the stream. Runs after the instruction finder
    and before the node persister, over the ``nodes`` channel."""

    def __init__(self, module: InstructionDistributor | None = None) -> None:
        self.module = module or InstructionDistributor()

    async def run(self, state: state.State) -> dict:
        """Distributes each lead-in's shared instruction onto the governed exercises."""
        nodes = state.get('nodes', [])
        cleaned = await distribute_instructions(nodes, module=self.module)
        return {'nodes': cleaned}
