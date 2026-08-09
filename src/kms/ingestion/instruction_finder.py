import asyncio
import logging

import dspy
from pydantic import BaseModel

from kms.core import logs, models, recording, state, walker

logger = logging.getLogger(__name__)
LOOKAHEAD_BUDGET = 2000
BACKWARD_CONTEXT_BUDGET = 200


class WindowNode(BaseModel):
    position: int
    type: str
    content: str | None = None


class Signature(dspy.Signature):
    r"""
    Find the exercise LEAD-IN nodes in a run of textbook nodes and return their
    positions.

    A lead-in is a short directive that introduces a run of OTHER exercises and
    states a shared instruction. The decisive test: a lead-in has NO reference
    number of its own, yet gives an imperative meant for a run of
    separately-numbered exercises that follow it. It may name an explicit RANGE
    ("In Exercises 1.23-1.25, find the eigenvalues of each matrix."; "For
    Exercises 5-6, determine whether the set is a subspace.") OR name no range
    at all — and the range-less form is the COMMON one, so do NOT require a
    range: "In the following exercises, simplify each expression.", "For the
    following exercises, find the gradient.", "Prove each of the following.",
    "9-16 Sketch the polar curve." are all lead-ins. Tag either form.

    CRITICAL: a node that BEGINS WITH ITS OWN EXERCISE NUMBER ("1.15 Perform
    each multiplication.", "✓ 1.17 For a homomorphism …", "1.22 Represent each
    linear map …") is an EXERCISE, never a lead-in — do NOT tag it, even though
    its own imperative reads like an instruction, because that imperative
    governs only that one exercise's own subparts. A lead-in governs several
    DIFFERENTLY-numbered exercises; an exercise governs only its own (a)(b)(c).
    Prose and section headers are never lead-ins either.

    Return the positions of the lead-in nodes, over the given nodes ONLY. The
    list may be empty.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position."
    )
    context_before: str | None = dspy.InputField(
        default=None,
        description=(
            'Optional text immediately before the window, in document '
            'order. CONTEXT ONLY — use it to place the lead-ins; never '
            'tag or copy nodes from it.'
        ),
    )
    instruction_positions: list[int] = dspy.OutputField(
        description='Positions of exercise lead-in nodes (shared-instruction directives).'
    )


class InstructionFinder(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        current_nodes: list[WindowNode],
        context_before: str | None = None,
    ) -> list[int]:
        result = await self.finder.acall(
            current_nodes=current_nodes,
            context_before=context_before or '',
        )
        if self._recorder:
            self._recorder.record(
                'instruction_finder',
                {'current_nodes': current_nodes},
                result,
            )
        positions = list(result.instruction_positions or [])
        logger.debug(
            'tag: %d nodes in, lead-in position(s) %s',
            len(current_nodes),
            positions or 'none',
        )
        return positions

    def forward(
        self,
        current_nodes: list[WindowNode],
        context_before: str | None = None,
    ) -> list[int]:
        return asyncio.run(
            self.aforward(current_nodes, context_before=context_before)
        )


async def tag_instructions(
    nodes: list[models.ASTNode],
    module: InstructionFinder,
    budget: int = LOOKAHEAD_BUDGET,
) -> list[models.ASTNode]:
    module = module
    if not nodes:
        return nodes
    cursor, node_count = 0, len(nodes)
    while cursor < node_count:
        end = walker.window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        positions = await module.aforward(
            [
                WindowNode(
                    position=position,
                    type=node.type,
                    content=node.content,
                )
                for position, node in enumerate(window)
            ],
            context_before=walker.content_before(
                nodes, cursor, BACKWARD_CONTEXT_BUDGET
            ),
        )
        for position in positions:
            clamped = min(max(position, 0), last_local)
            global_index = cursor + clamped
            original = nodes[global_index]
            nodes[global_index] = models.ASTNode(
                type='instruction',
                content=original.content,
                id=original.id,
                segment_index=original.segment_index,
            )
            logger.debug(
                'lead-in at node %s: %r',
                nodes[global_index].id,
                logs.elide(nodes[global_index].content),
            )
        cursor = end
    tagged = [node for node in nodes if node.type == 'instruction']
    logger.info(
        'instruction finder: %d node(s) -> %d lead-in(s) tagged',
        node_count,
        len(tagged),
    )
    return nodes


class InstructionFinderNode:
    def __init__(self, module: InstructionFinder) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        nodes = await tag_instructions(
            state.get('nodes', []), module=self.module
        )
        return {'nodes': nodes}

