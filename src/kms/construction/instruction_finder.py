"""Finds shared exercise instructions with boolean routing and growth."""

import logging

import dspy

from kms import config
from kms.core import (
    context_window,
    identity,
    models,
    module,
    recording,
    state,
)

logger = logging.getLogger(__name__)


class InstructionRouterSignature(dspy.Signature):
    r"""
    Classify only `target_node`. The before and after lists are context only.
    Return True only when the target node is an unnumbered directive
    introducing or governing multiple exercises. Return False for a numbered
    exercise, lettered fragment without its lead-in, ordinary prose, heading,
    answer, or continuation of an earlier instruction.

    Return only a boolean. Do not return a reason, label, span, or text.
    Answer only the boolean True or False.
    """

    context_before: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately before target_node; context only.',
    )
    target_node: models.NodeInput = dspy.InputField(
        description='The only one-based node record being classified as an instruction start.',
    )
    context_after: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately after target_node; context only.',
    )
    is_instruction_start: bool = dspy.OutputField(
        description='True only when the designated node starts a shared exercise instruction.'
    )


class InstructionGrowerSignature(dspy.Signature):
    r"""
    Classify only `candidate_node` as part of one shared exercise
    instruction. The accepted nodes define the instruction currently being
    grown.

    INCLUDE the candidate only when it is a grammatical fragment of the
    shared directive itself. Examples of continuations:

    - "find equations of" → "a. the tangent plane and"
    - "identify the extrema" → "using the diagram below"
    - "find equations of" → "a. the tangent plane and" → "b. the normal line"

    EXCLUDE the candidate when it starts, states, or supplies an individual
    exercise. An exercise may be numbered, lettered, unnumbered, imperative,
    interrogative, formula-only, or split across several nodes. Examples of
    exclusions:

    - "For the following exercises, simplify" → "925. 4 + 7"
    - "For the following exercises, find the gradient" →
      "Find the gradient of f(x, y)."
    - "For the following exercises, determine convergence" →
      "Does the series converge?"
    - "For the following exercises, find the prime factorization" →
      "420"

    The words "for the following exercises" announce the instruction; they do
    not make the following exercise text part of the instruction. A candidate
    that contains a specific exercise number, expression, quantity, point,
    equation, question, or requested operation is normally an exercise, not
    instruction continuation. Do not use numbering as the sole criterion:
    apply the same boundary to unlabeled tasks.

    Once a candidate is excluded as an exercise, return False. Never absorb
    that exercise or its continuation nodes into the shared instruction.
    Return only a boolean. Answer only True or False.
    """

    accepted_nodes: list[models.NodeInput] = dspy.InputField(
        description='Accepted one-based node records in document order; no assets or bytes.',
    )
    context_before: list[models.NodeInput] = dspy.InputField(
        description='One-based node records immediately before candidate_node; context only.',
    )
    candidate_node: models.NodeInput = dspy.InputField(
        description='The only one-based node record being classified for inclusion.',
    )
    context_after: list[models.NodeInput] = dspy.InputField(
        description='One-based node records immediately after candidate_node; context only.',
    )
    include_next_node: bool = dspy.OutputField(
        description='True only when candidate_node continues the anchored instruction.'
    )


def _instruction_input(
    node: context_window.ContextNode, local_index: int = 0
) -> models.NodeInput:
    return models.NodeInput(
        index=local_index + 1,
        node_type=node.type or '',
        text=node.content or '',
    )


def _instruction_lists(
    nodes: list[context_window.ContextNode],
) -> tuple[list[models.NodeInput], models.NodeInput, list[models.NodeInput]]:
    target_index = next(
        index for index, node in enumerate(nodes) if node.marker is not None
    )
    return (
        [
            _instruction_input(node, index)
            for index, node in enumerate(nodes[:target_index])
        ],
        _instruction_input(nodes[target_index]),
        [
            _instruction_input(node, index)
            for index, node in enumerate(nodes[target_index + 1 :])
        ],
    )


def _router_demo(
    text: str, is_instruction_start: bool, node_type: str = 'paragraph'
) -> dspy.Example:
    window = [
        context_window.ContextNode(
            position=0,
            type=node_type,
            marker='designated',
            content=text,
        )
    ]
    before, target, after = _instruction_lists(window)
    return dspy.Example(
        context_before=before,
        target_node=target,
        context_after=after,
        is_instruction_start=is_instruction_start,
    ).with_inputs('context_before', 'target_node', 'context_after')


def _grower_demo(
    accepted: list[str],
    candidate: str,
    include_next_node: bool,
    candidate_type: str = 'paragraph',
) -> dspy.Example:
    accepted_nodes = [
        models.NodeInput(index=index + 1, node_type='paragraph', text=text)
        for index, text in enumerate(accepted)
    ]
    candidate_context = [
        context_window.ContextNode(
            position=0,
            type=candidate_type,
            marker='candidate',
            content=candidate,
        )
    ]
    before, target, after = _instruction_lists(candidate_context)
    return dspy.Example(
        accepted_nodes=accepted_nodes,
        context_before=before,
        candidate_node=target,
        context_after=after,
        include_next_node=include_next_node,
    ).with_inputs(
        'accepted_nodes',
        'context_before',
        'candidate_node',
        'context_after',
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
            _router_demo(
                'For the following exercises, simplify each expression.',
                True,
            ),
            _router_demo(
                '185. Determine whether the series converges.',
                False,
            ),
            _router_demo(
                '12. Find the derivative of f(x).',
                False,
                node_type='list',
            ),
            _router_demo('a. Find the tangent plane.', False),
            _router_demo(
                'For Exercises 8–10, determine whether each set is a subspace.',
                True,
            ),
        ]

    def encode(
        self,
        context_before: list[models.NodeInput],
        target_node: models.NodeInput,
        context_after: list[models.NodeInput],
    ) -> dict[str, object]:
        """Passes document-ordered target and context to DSPy."""
        return {
            'context_before': context_before,
            'target_node': target_node,
            'context_after': context_after,
        }

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
            _grower_demo(
                ['For the following exercises, find equations of:'],
                'a. the tangent plane and',
                True,
            ),
            _grower_demo(
                [
                    'For the following exercises, find equations of:',
                    'a. the tangent plane and',
                ],
                'b. the normal line to the surface.',
                True,
            ),
            _grower_demo(
                [
                    'For the following exercises, find equations of:',
                    'a. the tangent plane and',
                ],
                '302. $z = 4x^2 + y^2$, point P(2, 1, 8)',
                False,
            ),
            _grower_demo(
                ['For the following exercises, simplify each expression.'],
                'For Exercises 8–10, determine whether each set is a subspace.',
                False,
            ),
            _grower_demo(
                [
                    'For the following exercises, find the gradient.',
                    '280. Find the gradient of f(x, y).',
                ],
                'Use the definition of the gradient.',
                False,
            ),
            _grower_demo(
                ['For the following exercises, find the gradient.'],
                'Find the gradient of f(x, y).',
                False,
            ),
            _grower_demo(
                ['For the following exercises, determine convergence.'],
                'Does the series converge?',
                False,
            ),
            _grower_demo(
                ['For the following exercises, identify the extrema.'],
                'Decorative publisher illustration.',
                False,
                candidate_type='image',
            ),
            _grower_demo(
                [
                    'For the following exercises, use the diagram to identify the extrema.'
                ],
                'Diagram of the curve and its marked extrema.',
                True,
                candidate_type='image',
            ),
        ]

    def encode(
        self,
        accepted_nodes: list[models.NodeInput],
        context_before: list[models.NodeInput],
        candidate_node: models.NodeInput,
        context_after: list[models.NodeInput],
    ) -> dict[str, object]:
        """Passes document-ordered target and context to DSPy."""
        return {
            'accepted_nodes': accepted_nodes,
            'context_before': context_before,
            'candidate_node': candidate_node,
            'context_after': context_after,
        }

    def decode(self, prediction, **inputs) -> bool:
        """Returns a strictly boolean growth decision."""
        return _strict_bool(prediction.include_next_node, 'include_next_node')


def _strict_bool(value: object, field_name: str) -> bool:
    """Rejects model values that are not actual booleans."""
    return module.require_bool(value, field_name)


async def find_instruction_spans(
    nodes: list[models.SourceNode],
    router: InstructionRouter,
    grower: InstructionGrower,
) -> list[list[int]]:
    """Scans and grows instruction spans deterministically over ``nodes``."""
    spans: list[list[int]] = []
    cursor = 0
    finder_settings = config.get_settings().stages.finders
    context_budget = finder_settings.instruction_finder.context_budget
    max_span_budget = finder_settings.instruction_finder.max_span_budget

    while cursor < len(nodes):
        selected = context_window.select_around(
            nodes,
            [cursor],
            backward_budget=0,
            forward_budget=context_budget,
            marker='designated',
        )
        before, target, after = _instruction_lists(selected)
        is_start = _strict_bool(
            await router.aforward(
                context_before=before,
                target_node=target,
                context_after=after,
            ),
            'is_instruction_start',
        )
        if not is_start:
            cursor += 1
            continue

        end = cursor + 1
        accepted_size = context_window.estimate_tokens(nodes[cursor])
        while end < len(nodes):
            next_size = context_window.estimate_tokens(nodes[end])
            if accepted_size + next_size > max_span_budget:
                raise ValueError(
                    f'instruction span reaches the {max_span_budget}-token '
                    f'look-ahead limit at cursor {cursor}'
                )
            accepted = [
                _instruction_input(node, index)
                for index, node in enumerate(
                    context_window.project_nodes(nodes[cursor:end])
                )
            ]
            selected_candidate = context_window.select_around(
                nodes,
                [end],
                backward_budget=0,
                forward_budget=context_budget,
                marker='candidate',
            )
            candidate_before, candidate, candidate_after = _instruction_lists(
                selected_candidate
            )
            include = _strict_bool(
                await grower.aforward(
                    accepted_nodes=accepted,
                    context_before=candidate_before,
                    candidate_node=candidate,
                    context_after=candidate_after,
                ),
                'include_next_node',
            )
            if not include:
                break
            accepted_size += next_size
            end += 1

        member_positions = list(range(cursor, end))
        spans.append(member_positions)
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
        source = state['source'].key
        if not source:
            raise ValueError('instruction detection requires a source')
        instructions = [
            models.Instruction(block=list(span), member_positions=list(span))
            for span in spans
        ]
        identity.assign_instruction_ids(instructions, source)
        return {'instructions': instructions}
