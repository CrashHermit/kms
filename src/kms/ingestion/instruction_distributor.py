r"""
Instruction distributor — records which exercise nodes a grouped-exercise
lead-in governs, as a hub beside the stream, then removes the lead-in node.

The instruction finder has already stamped ``NodeType.INSTRUCTION`` on every
lead-in node. This pass walks the stream, anchors on each INSTRUCTION node, and
asks the LLM which of the following exercise nodes it governs. The answer
becomes one ``models.Instruction`` hub over those node ids. The INSTRUCTION
node is then removed from the stream — its sentence lives on the hub.

The hub sits BESIDE the nodes, like ``Statement`` and ``Procedure``, and for
the same reason. This pass used to prepend the directive onto each governed
node's ``content``, which put two kinds of wrong text into the provenance
layer: the same sentence repeated once per governed exercise (55% of all
``:Node.content`` written on a three-page fixture, 42% of it redundant), and,
worse, a sentence the page does not contain — the model returns a normalised
imperative ("simplify") where the page reads "In the following exercises,
simplify." A ``:Node`` is defined as one verbatim block of the page, so the
synthesized form could not live there. It is now ``directive`` on the hub,
beside ``text``, which is the lead-in exactly as printed.

The cost of the move is that a governed node no longer carries its directive
in its own text: anything reading one node alone — an embedding, a window of
context — sees the bare exercise and must traverse ``:GOVERNS`` for the
instruction.

The extent is judged by the LLM, not by numbers. A lead-in may name a range ("In
Exercises 1.23-1.25, …") or may not ("Answer the following."), so a range parser
can't decide who is governed — the model reads the following problems and
decides where the governed run ends. It uses the SAME growing look-ahead pattern
as the finders and the splitter:

  * Anchor on an INSTRUCTION node. Its candidates are the non-instruction nodes
    that follow it, up to the next INSTRUCTION (a new lead-in starts a new
    governance).
  * Take a look-ahead window of those candidates (whole nodes up to a token
    budget) and ask the LLM which of them the lead-in governs, plus the shared
    instruction to apply.
  * If the governed run reaches the window's edge it may continue, so GROW the
    window (double the budget) and re-read; if a non-governed node follows the
    run (bounded), or the candidates are exhausted, BANK — prepend the
    instruction to each governed node.
  * After all governance is resolved, delete every INSTRUCTION node from the
    stream.

Runs BEFORE the node persister — instruction nodes are never written to Neo4j,
and the persisted stream carries exercises with their instruction already
prepended to content.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import logs, models, state, walker
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

# Same growing look-ahead shape as the finders/splitter (~4 chars/token).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowProblem(BaseModel):
    """One following node as the LLM sees it: position and content."""

    position: int
    content: str | None = None


class GovernExtent(dspy.Signature):
    r"""
    Given an exercise LEAD-IN and the exercise nodes that FOLLOW it in document
    order, decide which of those exercises the lead-in's shared instruction
    governs, and give the instruction to apply.

    A lead-in states one imperative for a run of exercises. Some name a range
    ("In Exercises 1.23-1.25, find the eigenvalues of each matrix."), some do
    not ("Answer the following.", "Prove each of the following statements.").
    Judge governance by MEANING, not by numbers: the governed exercises are a
    contiguous run that STARTS at the first exercise after the lead-in and
    continues while the lead-in's instruction still sensibly applies to them,
    and STOPS when it no longer does — an exercise that is clearly a different
    kind of task, or the start of a different group.

    Each following-problem entry contains the exercise's raw text (the exercise
    number is in the text itself). Judge governance from the text content — do
    not require a separate number field.

    Return:
      * instruction — the shared imperative to apply to the governed exercises,
        copied as written but WITHOUT any "In Exercises X-Y," framing ("find the
        eigenvalues of each matrix", "answer the following"). EMPTY string if
        the lead-in actually governs nothing here.
      * governed_positions — the `position` values of the governed exercises (a
        run from the first). EMPTY list if none are governed.
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
    """Determines which following exercises a lead-in governs.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(GovernExtent)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self, lead_in: str, following: list[WindowProblem]
    ) -> tuple[str, list[int]]:
        """Judge one lead-in's extent.

        Args:
            lead_in: The lead-in node's text.
            following: The candidate exercises, each with a local position.

        Returns:
            The shared instruction and the window-local positions it governs.
        """
        result = await self.judge.acall(
            lead_in=lead_in, following_problems=following
        )
        if self._recorder:
            self._recorder.record(
                'instruction_distributor',
                {'lead_in': lead_in, 'following_problems': following},
                result,
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

    def forward(
        self, lead_in: str, following: list[WindowProblem]
    ) -> tuple[str, list[int]]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(lead_in, following))


def _node_text(node: models.ASTNode) -> str:
    """The node's content as the LLM should see it.

    Args:
        node: The node to render.

    Returns:
        Its content, stripped, or the empty string.
    """
    return (node.content or '').strip()


def _window(
    candidates: list[models.ASTNode], budget: int
) -> list[models.ASTNode]:
    """The candidates that fit in one look-ahead window.

    Args:
        candidates: The following nodes, in document order.
        budget: The soft token budget for the window.

    Returns:
        Whole nodes up to the budget, always at least one.
    """
    window, accumulated = [], 0
    for node in candidates:
        token_count = walker.estimate_tokens(node)
        if window and accumulated + token_count > budget:
            break
        window.append(node)
        accumulated += token_count
    return window


async def _govern_one(
    lead_in: models.ASTNode,
    candidates: list[models.ASTNode],
    module: InstructionDistributor,
) -> models.Instruction | None:
    """Growing-window walk for one lead-in.

    Finds the governed run among the following exercise nodes and records it
    as a hub over their ids. Nothing is written onto the governed nodes: they
    are verbatim page blocks and the directive is not part of what they say.

    Args:
        lead_in: The lead-in node.
        candidates: The exercise nodes that follow it, in document order.
        module: The governance module.

    Returns:
        The hub, or None when this lead-in governs nothing.
    """
    if not candidates:
        return None
    size = LOOKAHEAD_BUDGET
    while True:
        window = _window(candidates, size)
        last_local = len(window) - 1
        exhausted = len(window) == len(candidates)

        instruction, positions = await module.aforward(
            lead_in=lead_in.content or '',
            following=[
                WindowProblem(position=position, content=_node_text(node))
                for position, node in enumerate(window)
            ],
        )
        governed = sorted(
            {min(max(position, 0), last_local) for position in positions}
        )

        if not governed:
            # This lead-in governs nothing here.
            return None
        run_end = governed[-1]

        if exhausted or size >= MAX_LOOKAHEAD_BUDGET or run_end < last_local:
            # Bounded or nothing left to gather: bank.
            members = [
                window[position].id
                for position in governed
                if window[position].id is not None
            ]
            if not members or lead_in.id is None:
                return None
            return models.Instruction(
                node_id=lead_in.id,
                text=lead_in.content or '',
                directive=instruction or None,
                members=members,
            )
        # The run reaches the window edge — grow and re-read.
        size *= 2


async def distribute_instructions(
    nodes: list[models.ASTNode],
    module: InstructionDistributor,
) -> tuple[list[models.ASTNode], list[models.Instruction]]:
    """Resolve every lead-in's governance over the exercises that follow it.

    Args:
        nodes: The flat node stream, with INSTRUCTION nodes already tagged.
        module: The governance module.

    Returns:
        The cleaned node stream — instruction nodes removed, every other node
        untouched — and one ``Instruction`` hub per lead-in that governs
        anything, in document order.
    """
    lead_ins = [
        node for node in nodes if isinstance(node, models.InstructionNode)
    ]
    if not lead_ins:
        logger.info(
            'instruction distributor: no-op (0 lead-in(s), %d node(s))',
            len(nodes),
        )
        return nodes, []

    module = module

    # Index the stream so we know what follows each lead-in.
    position_of = {
        node.id: position
        for position, node in enumerate(nodes)
        if node.id is not None
    }
    lead_positions = sorted(
        position_of.get(node.id) for node in lead_ins if node.id is not None
    )
    stream_end = len(nodes)
    instructions: list[models.Instruction] = []

    for node in lead_ins:
        here = position_of.get(node.id)
        if here is None:
            continue
        # Candidates end at the next lead-in or the end of the stream.
        next_lead = min(
            (
                position
                for position in lead_positions
                if position is not None and position > here
            ),
            default=stream_end,
        )
        candidates = [
            candidate
            for candidate in nodes[here + 1 : next_lead]
            if not isinstance(candidate, models.InstructionNode)
        ]
        hub = await _govern_one(node, candidates, module)
        if hub is not None:
            instructions.append(hub)

    # Remove instruction nodes from the stream. The lead-in's own sentence is
    # not lost with them — it is on its hub, verbatim, stored once.
    cleaned = [
        node for node in nodes if not isinstance(node, models.InstructionNode)
    ]
    logger.info(
        'instruction distributor: %d lead-in(s) removed, %d hub(s) over %d '
        'governed node(s), %d of %d node(s) remain',
        len(lead_ins),
        len(instructions),
        sum(len(hub.members) for hub in instructions),
        len(cleaned),
        len(nodes),
    )
    return cleaned, instructions


# --- LangGraph node ---


class InstructionDistributorNode:
    """Distributes lead-in directives, then drops the instruction nodes.

    Runs after the instruction finder and before the node persister, over the
    ``nodes`` channel.

    Args:
        module: The governance module.
    """

    def __init__(self, module: InstructionDistributor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Distribute each lead-in's instruction onto its exercises.

        Args:
            state: The pipeline state, holding the flat node stream.

        Returns:
            The cleaned `nodes` channel and the `instructions` overlay.
        """
        nodes = state.get('nodes', [])
        cleaned, instructions = await distribute_instructions(
            nodes, module=self.module
        )
        return {'nodes': cleaned, 'instructions': instructions}
