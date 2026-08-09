import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import logs, models, recording, state, walker

logger = logging.getLogger(__name__)
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowProblem(BaseModel):
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
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(GovernExtent)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self, lead_in: str, following: list[WindowProblem]
    ) -> tuple[str, list[int]]:
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
            result.instruction.strip(),
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
        return asyncio.run(self.aforward(lead_in, following))


def _node_text(node: models.ASTNode) -> str:
    return (node.content or '').strip()


def _window(
    candidates: list[models.ASTNode], budget: int
) -> list[models.ASTNode]:
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
            return None
        run_end = governed[-1]

        if exhausted or size >= MAX_LOOKAHEAD_BUDGET or run_end < last_local:
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
        size *= 2


async def distribute_instructions(
    nodes: list[models.ASTNode],
    module: InstructionDistributor,
) -> tuple[list[models.ASTNode], list[models.Instruction]]:
    lead_ins = [node for node in nodes if node.type == 'instruction']
    if not lead_ins:
        logger.info(
            'instruction distributor: no-op (0 lead-in(s), %d node(s))',
            len(nodes),
        )
        return nodes, []

    module = module
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
            if not candidate.type == 'instruction'
        ]
        hub = await _govern_one(node, candidates, module)
        if hub is not None:
            instructions.append(hub)
    cleaned = [node for node in nodes if node.type != 'instruction']
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


class InstructionDistributorNode:
    def __init__(self, module: InstructionDistributor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        nodes = state.get('nodes', [])
        cleaned, instructions = await distribute_instructions(
            nodes, module=self.module
        )
        return {'nodes': cleaned, 'instructions': instructions}

