"""Finds shared exercise instructions with boolean routing and growth."""

import logging

import dspy

from kms import config
from kms.core import content, identity, models, module, recording, state, walker

logger = logging.getLogger(__name__)


class InstructionRouterSignature(dspy.Signature):
    r"""
    Classify only the node marked `<designated>`.

    Return True only when that node is an unnumbered directive introducing or
    governing multiple exercises. Return False for a numbered exercise,
    lettered fragment without its lead-in, ordinary prose, heading, answer, or
    continuation of an earlier instruction. Surrounding nodes are context
    only; adjacency does not make the designated node an instruction.

    Return only a boolean. Do not return a reason, label, span, or text.
    Answer only the boolean True or False.
    """

    current_node: content.ContentParts = dspy.InputField(
        description=(
            'A bounded local window. The designated node is labeled '
            '`<designated>`; surrounding nodes are context only.'
        )
    )
    is_instruction_start: bool = dspy.OutputField(
        description='True only when the designated node starts a shared exercise instruction.'
    )


class InstructionGrowerSignature(dspy.Signature):
    r"""
    Classify only the node marked `<candidate>`.

    `instruction_nodes` are the accepted members of one instruction. The
    candidate owns the inclusion decision: return True only when it continues
    that same directive and completes its content. Lettered fragments may
    continue it before the first numbered exercise. Return False for the first
    numbered exercise, a new instruction, unrelated prose, or adjacency alone;
    never resume an instruction after an exercise.

    Return only a boolean. Do not return a reason, label, span, or text.
    Answer only the boolean True or False.
    """

    instruction_nodes: content.ContentParts = dspy.InputField(
        description='Accepted instruction nodes in document order.'
    )
    next_node: content.ContentParts = dspy.InputField(
        description=(
            'A bounded local window around the immediately following '
            'candidate, labeled `<candidate>`.'
        )
    )
    include_next_node: bool = dspy.OutputField(
        description='True only when next_node continues the anchored instruction.'
    )


class InstructionRouter(module.Module):
    """Routes each node to either instruction-start or ordinary content."""

    signature = InstructionRouterSignature
    record_name = 'instruction_router'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            # A complete, unnumbered directive is a start.
            dspy.Example(
                current_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            marker='designated',
                            content='For the following exercises, simplify each expression.',
                        )
                    ]
                ),
                is_instruction_start=True,
            ).with_inputs('current_node'),
            # The exercise number is the boundary, even when its wording is
            # an imperative.
            dspy.Example(
                current_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            marker='designated',
                            content='185. Determine whether the series converges.',
                        )
                    ]
                ),
                is_instruction_start=False,
            ).with_inputs('current_node'),
            dspy.Example(
                current_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='list',
                            marker='designated',
                            content='12. Find the derivative of f(x).',
                        )
                    ]
                ),
                is_instruction_start=False,
            ).with_inputs('current_node'),
            # A lettered node without its shared lead-in is not an instruction
            # start.
            dspy.Example(
                current_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            marker='designated',
                            content='a. Find the tangent plane.',
                        )
                    ]
                ),
                is_instruction_start=False,
            ).with_inputs('current_node'),
            dspy.Example(
                current_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            marker='designated',
                            content='For Exercises 8–10, determine whether each set is a subspace.',
                        )
                    ]
                ),
                is_instruction_start=True,
            ).with_inputs('current_node'),
        ]

    def encode(self, current_node: list[walker.WindowNode]) -> dict:
        """Builds the multimodal input for one designated node window."""
        return {'current_node': content.labeled_content_parts(current_node)}

    def decode(self, prediction, **inputs) -> bool:
        """Returns a strictly boolean routing decision."""
        return _strict_bool(
            prediction.is_instruction_start, 'is_instruction_start'
        )


class InstructionGrower(module.Module):
    """Grows one routed instruction by considering one candidate at a time."""

    signature = InstructionGrowerSignature
    record_name = 'instruction_grower'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            # Lettered fragments can complete a split lead-in.
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, find equations of:',
                        )
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            marker='candidate',
                            content='a. the tangent plane and',
                        )
                    ]
                ),
                include_next_node=True,
            ).with_inputs('instruction_nodes', 'next_node'),
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
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
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=2,
                            type='paragraph',
                            marker='candidate',
                            content='b. the normal line to the surface.',
                        )
                    ]
                ),
                include_next_node=True,
            ).with_inputs('instruction_nodes', 'next_node'),
            # The first numbered exercise ends the instruction.
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
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
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=2,
                            type='paragraph',
                            marker='candidate',
                            content='302. $z = 4x^2 + y^2$, point P(2, 1, 8)',
                        )
                    ]
                ),
                include_next_node=False,
            ).with_inputs('instruction_nodes', 'next_node'),
            # A new directive is a boundary, not a continuation.
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, simplify each expression.',
                        )
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            marker='candidate',
                            content='For Exercises 8–10, determine whether each set is a subspace.',
                        )
                    ]
                ),
                include_next_node=False,
            ).with_inputs('instruction_nodes', 'next_node'),
            # A continuation after an exercise number cannot be pulled back
            # into the earlier instruction.
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, find the gradient.',
                        ),
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            content='280. Find the gradient of f(x, y).',
                        ),
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=2,
                            type='paragraph',
                            marker='candidate',
                            content='Use the definition of the gradient.',
                        )
                    ]
                ),
                include_next_node=False,
            ).with_inputs('instruction_nodes', 'next_node'),
            # Positional adjacency alone does not make an image a member.
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, identify the extrema.',
                        )
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=1,
                            type='image',
                            marker='candidate',
                            content='Decorative publisher illustration.',
                        )
                    ]
                ),
                include_next_node=False,
            ).with_inputs('instruction_nodes', 'next_node'),
            # An image whose visible content is explicitly used by the
            # directive belongs to the instruction span.
            dspy.Example(
                instruction_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, use the diagram to identify the extrema.',
                        )
                    ]
                ),
                next_node=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=1,
                            type='image',
                            marker='candidate',
                            content='Diagram of the curve and its marked extrema.',
                        )
                    ]
                ),
                include_next_node=True,
            ).with_inputs('instruction_nodes', 'next_node'),
        ]

    def encode(
        self,
        instruction_nodes: list[walker.WindowNode],
        next_node: list[walker.WindowNode],
    ) -> dict:
        """Builds labelled multimodal inputs for the anchor and candidate windows."""
        return {
            'instruction_nodes': content.labeled_content_parts(
                instruction_nodes
            ),
            'next_node': content.labeled_content_parts(next_node),
        }

    def decode(self, prediction, **inputs) -> bool:
        """Returns a strictly boolean growth decision."""
        return _strict_bool(prediction.include_next_node, 'include_next_node')


def _strict_bool(value: object, field_name: str) -> bool:
    """Rejects model values that are not actual booleans."""
    if type(value) is not bool:
        raise ValueError(
            f'{field_name} must be a boolean, got {type(value).__name__}'
        )
    return value


def _node_view(
    node: models.Node, position: int, marker: str | None = None
) -> walker.WindowNode:
    """Builds a local view while retaining the stream position and id."""
    return walker.WindowNode(
        position=position,
        id=node.id,
        type=node.type,
        content=node.content,
        image_path=node.image_path,
        marker=marker,
    )


async def find_instruction_spans(
    nodes: list[models.Node],
    router: InstructionRouter,
    grower: InstructionGrower,
) -> list[list[int]]:
    """Scans and grows instruction spans deterministically over ``nodes``.

    The router is called once for each unclaimed node. A routed start anchors a
    span, and the grower then examines each following node exactly once until
    it returns False or the stream ends. Model positions are local and are
    converted to the stable node ids only after the decisions are validated.
    """
    spans: list[list[int]] = []
    cursor = 0
    context_budget = config.get_settings().stages.finders.instruction_context_budget

    while cursor < len(nodes):
        start_window = walker.marked_window_around(
            nodes, cursor, context_budget, 'designated'
        )
        is_start = _strict_bool(
            await router.aforward(current_node=start_window),
            'is_instruction_start',
        )
        if not is_start:
            cursor += 1
            continue

        end = cursor + 1
        while end < len(nodes):
            accepted = [
                _node_view(nodes[position], position)
                for position in range(cursor, end)
            ]
            candidate_window = walker.marked_window_around(
                nodes, end, context_budget, 'candidate'
            )
            include = _strict_bool(
                await grower.aforward(
                    instruction_nodes=accepted,
                    next_node=candidate_window,
                ),
                'include_next_node',
            )
            if not include:
                break
            end += 1

        member_ids = [nodes[position].id for position in range(cursor, end)]
        if any(node_id is None for node_id in member_ids):
            raise ValueError(
                f'instruction span at positions {cursor}:{end} references '
                'a node without a stable id'
            )
        spans.append([node_id for node_id in member_ids if node_id is not None])
        cursor = end

    logger.info(
        'instruction finder: %d nodes -> %d instruction span(s)',
        len(nodes),
        len(spans),
    )
    return spans


class InstructionFinderNode:
    """Graph node that turns routed instruction spans into models."""

    def __init__(
        self,
        router: InstructionRouter,
        grower: InstructionGrower,
    ) -> None:
        self.router = router
        self.grower = grower

    async def run(self, state: state.State) -> dict:
        """Finds instructions without mutating the canonical node stream."""
        spans = await find_instruction_spans(
            state.get('nodes', []), router=self.router, grower=self.grower
        )
        source = state.get('source_key', '').strip()
        if not source:
            source = models.source_key(state.get('source')) or ''
        if not source:
            raise ValueError('instruction detection requires a source')
        instructions = [
            models.Instruction(block=list(span), members=list(span))
            for span in spans
        ]
        identity.assign_instruction_ids(instructions, source)
        return {'instructions': instructions}
