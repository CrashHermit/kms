"""Finds shared exercise lead-in (instruction) sections in node runs."""

import logging

import dspy

from kms import config
from kms.core import content, models, module, recording, state, walker

logger = logging.getLogger(__name__)


class Signature(dspy.Signature):
    r"""
    Find shared exercise INSTRUCTIONS in the node run and return each one as
    an inclusive span of local `[position]` values.

    This is a boundary-finding task, not a classification or fact-extraction
    task. Find a short directive that introduces OTHER exercises, for example:

    - "In the following exercises, simplify each expression."
    - "For Exercises 5–6, determine whether the set is a subspace."
    - "For the following exercises, find equations of:"

    A shared instruction has no exercise number of its own. A node beginning
    with a problem number, such as "185.", "1.15", or "✓ 1.17", is the
    exercise it introduces or describes, never part of the shared instruction.
    Do not mistake an expression, answer request, or numbered problem for a
    shared lead-in.

    EXTENT:
    - Start at the first node of the shared directive.
    - Include consecutive continuation nodes that complete that directive,
      including lettered parts such as "a. ... b. ...".
    - Stop immediately before the first numbered exercise governed by it.
    - If the directive is complete in one node, emit a one-node span.
    - If OCR has split one directive across adjacent unnumbered fragments,
      keep those fragments together only when they clearly complete the same
      instruction. Do not invent missing text or absorb an unrelated exercise.
    - Section headings, ordinary prose, isolated answer choices, and numbered
      exercises are not instruction spans.

    OUTPUT:
    Emit every shared-instruction span in document order, using positions from
    this window only. Spans must be non-overlapping and inclusive. Return an
    empty list when no shared instruction is present. Do not emit anything
    merely because a node contains imperative wording if it has its own
    exercise number.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The look-ahead window's nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself. Emit spans '
            'over the `[position]` values only.'
        )
    )
    instruction_spans: list[walker.Span] = dspy.OutputField(
        description='Every exercise lead-in section found in current_nodes, as position spans, in '
        'document order. Empty list if none.'
    )


class InstructionFinder(module.Module):
    """Finds lead-in sections that introduce runs of exercises."""

    signature = Signature
    record_name = 'instruction_finder'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, find the gradient.',
                        ),
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            content='280. Find the gradient of $f(x, y) = x^2 + y^2$.',
                        ),
                        walker.WindowNode(
                            position=2,
                            type='paragraph',
                            content='281. Find the gradient of $f(x, y) = xy$.',
                        ),
                    ]
                ),
                instruction_spans=[walker.Span(start=0, end=0)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='In the following exercises, simplify each expression.',
                        ),
                        walker.WindowNode(
                            position=1,
                            type='list',
                            content='979. $17a + 9a$',
                        ),
                        walker.WindowNode(
                            position=2,
                            type='list',
                            content='980. $18z + 9z$',
                        ),
                    ]
                ),
                instruction_spans=[walker.Span(start=0, end=0)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, find equations of:',
                        ),
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            content='a. the tangent plane and',
                        ),
                        walker.WindowNode(
                            position=2,
                            type='paragraph',
                            content='b. the normal line to the given surface at the given point.',
                        ),
                        walker.WindowNode(
                            position=3,
                            type='paragraph',
                            content='302. $z = 4x^2 + y^2$, point $P(2, 1, 8)$',
                        ),
                    ]
                ),
                instruction_spans=[walker.Span(start=0, end=2)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='282. Find the gradient of $f(x, y, z)$ at $P$ and the directional derivative in the direction of $\\mathbf{u}$.',
                        ),
                    ]
                ),
                instruction_spans=[],
            ).with_inputs('current_nodes'),
        ]

    def encode(self, current_nodes: list[walker.WindowNode]) -> dict:
        """Builds the finder-signature kwargs for one window."""
        return {'current_nodes': content.labeled_content_parts(current_nodes)}

    def decode(self, prediction, **inputs) -> list[walker.Span]:
        """Returns the lead-in spans from the prediction."""
        return module.as_list(prediction.instruction_spans)


async def find_instruction_spans(
    nodes: list[models.ASTNode],
    module: InstructionFinder,
    budget: int | None = None,
    max_budget: int | None = None,
) -> list[list[int]]:
    """Finds all lead-in spans across the whole node stream.

    Args:
        nodes: The node stream, with ids assigned.
        module: The finder module.
        budget: Initial look-ahead token budget per window.
        max_budget: Cap on window growth before banking as-is.

    Returns:
        A list of member-id lists, one per lead-in section.
    """
    if budget is None:
        budget = config.get_settings().stages.finders.lookahead_budget
    if max_budget is None:
        max_budget = config.get_settings().stages.finders.max_lookahead_budget
    spans = await walker.find_spans(nodes, module, budget, max_budget)
    logger.info(
        'instruction finder: %d nodes -> %d lead-in span(s)',
        len(nodes),
        len(spans),
    )
    return spans


class InstructionFinderNode:
    """Graph node that turns lead-in spans into Instruction models."""

    def __init__(self, module: InstructionFinder) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Finds instructions in the state's nodes.

        Args:
            state: Pipeline state with nodes.

        Returns:
            An ``instructions`` update for the state.
        """
        spans = await find_instruction_spans(
            state.get('nodes', []), module=self.module
        )
        instructions = [
            models.Instruction(block=list(span), members=list(span))
            for span in spans
        ]
        return {'instructions': instructions}
