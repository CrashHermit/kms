"""Finds pedagogical units with sequential boolean routers."""

import logging

import dspy

from kms import config
from kms.core import context_window, models, module, recording, state

logger = logging.getLogger(__name__)


class PedagogicalStartRouterSignature(dspy.Signature):
    r"""
    Classify only `target_node` as the possible start of one pedagogical unit.
    The before and after lists are context only. Return True when the target
    starts a labelled or numbered definition, theorem, proposition, lemma,
    corollary, axiom, law, model, rule, principle, worked example, exercise,
    problem, or prescribed procedure. A procedure lead-in and its numbered
    steps form one unit.

    Return False for headings, ordinary narrative, shared exercise
    instructions, continuations, dangling OCR fragments, isolated answer
    choices, formatting artifacts, and material already owned by an earlier
    unit. Do not merge distinct base numbers or attach unrelated nearby text.

    Return only a boolean. Answer only True or False.
    """

    context_before: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately before target_node; context only.'
    )
    target_node: models.NodeInput = dspy.InputField(
        description='The only one-based node record being classified as a pedagogical unit start.'
    )
    context_after: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately after target_node; context only.'
    )
    is_pedagogical_start: bool = dspy.OutputField(
        description='True only when target_node starts one pedagogical unit.'
    )


class PedagogicalEndRouterSignature(dspy.Signature):
    r"""
    Classify only `candidate_node` as the final node of the pedagogical unit
    anchored by `start_node`. The before and after lists are context only.
    Return True when the candidate completes the anchored definition, theorem,
    proposition, lemma, corollary, axiom, law, model, rule, principle, worked
    example, exercise, problem, or prescribed procedure. Return True for a
    one-node unit when candidate_node is the same node as start_node.

    Return False when the anchored unit clearly continues beyond the candidate,
    including a proof, solution, derivation, calculation, computation output,
    subpart sequence, procedure step, or OCR continuation. Return False when
    candidate_node starts the next distinct unit or is unrelated prose. Do not
    merge distinct base numbers.

    Classify only candidate_node; do not emit a span or endpoint. Return only a
    boolean. Answer only True or False.
    """

    start_node: models.NodeInput = dspy.InputField(
        description='The fixed one-based node record where this pedagogical unit starts.'
    )
    context_before: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately before candidate_node; context only.'
    )
    candidate_node: models.NodeInput = dspy.InputField(
        description='The only one-based node record being classified as the possible unit end.'
    )
    context_after: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately after candidate_node; context only.'
    )
    is_pedagogical_end: bool = dspy.OutputField(
        description='True only when candidate_node is the final node of the anchored unit.'
    )


def _router_demo(
    *,
    target: str,
    decision: bool,
    before: list[str] | None = None,
    after: list[str] | None = None,
    node_type: str = 'paragraph',
) -> dspy.Example:
    context_before = [
        models.NodeInput(index=index + 1, node_type=node_type, text=text)
        for index, text in enumerate(before or [])
    ]
    target_node = models.NodeInput(index=1, node_type=node_type, text=target)
    context_after = [
        models.NodeInput(index=index + 1, node_type=node_type, text=text)
        for index, text in enumerate(after or [])
    ]
    return dspy.Example(
        context_before=context_before,
        target_node=target_node,
        context_after=context_after,
        is_pedagogical_start=decision,
    ).with_inputs('context_before', 'target_node', 'context_after')


def _end_demo(
    *,
    start: str,
    candidate: str,
    decision: bool,
    before: list[str] | None = None,
    after: list[str] | None = None,
) -> dspy.Example:
    return dspy.Example(
        start_node=models.NodeInput(index=1, node_type='paragraph', text=start),
        context_before=[
            models.NodeInput(index=index + 1, node_type='paragraph', text=text)
            for index, text in enumerate(before or [])
        ],
        candidate_node=models.NodeInput(
            index=1, node_type='paragraph', text=candidate
        ),
        context_after=[
            models.NodeInput(index=index + 1, node_type='paragraph', text=text)
            for index, text in enumerate(after or [])
        ],
        is_pedagogical_end=decision,
    ).with_inputs(
        'start_node', 'context_before', 'candidate_node', 'context_after'
    )


class PedagogicalStartRouter(module.Module):
    """Routes unowned nodes to pedagogical-unit starts."""

    signature = PedagogicalStartRouterSignature
    record_name = 'pedagogical_start_router'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            _router_demo(
                target="Exercise 1.2.1: Sketch the slope field for $y' = e^{x-y}$.",
                decision=True,
            ),
            _router_demo(
                target='For Exercises 1.2.1–1.2.3, answer each question.',
                decision=False,
            ),
            _router_demo(
                target='Proof.', decision=False, before=['Theorem 2.1.10.']
            ),
            _router_demo(
                target='Procedure: compute the determinant.',
                decision=True,
                after=['1. Form the matrix.', '2. Expand along the first row.'],
            ),
            _router_demo(
                target="Exercise 1.2.2: Sketch the slope field for $y' = x^2$.",
                decision=True,
                before=['Exercise 1.2.1: Sketch the slope field.'],
            ),
        ]

    def encode(
        self,
        context_before: list[models.NodeInput],
        target_node: models.NodeInput,
        context_after: list[models.NodeInput],
    ) -> dict[str, object]:
        """Passes typed target context unchanged to DSPy."""
        return {
            'context_before': context_before,
            'target_node': target_node,
            'context_after': context_after,
        }

    def decode(self, prediction: dspy.Prediction, **inputs: object) -> bool:
        """Returns a strictly boolean start decision."""
        return module.require_bool(
            prediction.is_pedagogical_start, 'is_pedagogical_start'
        )


class PedagogicalEndRouter(module.Module):
    """Routes candidate nodes to ends of anchored pedagogical units."""

    signature = PedagogicalEndRouterSignature
    record_name = 'pedagogical_end_router'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            _end_demo(
                start='Exercise 1.2.1: Sketch the slope field.',
                candidate='Exercise 1.2.1: Sketch the slope field.',
                decision=True,
            ),
            _end_demo(
                start='Exercise 1.2.7: Let $\\{x_n\\}$ be a sequence.',
                candidate='a) Show that $\\lim x_n = 0$.',
                decision=False,
                after=['b) Find an example where $\\{x_n\\}$ diverges.'],
            ),
            _end_demo(
                start='Exercise 1.2.7: Let $\\{x_n\\}$ be a sequence.',
                candidate='b) Find an example where $\\{x_n\\}$ diverges.',
                decision=True,
            ),
            _end_demo(
                start='Theorem 2.1.10. Every bounded monotone sequence converges.',
                candidate='Proof. Assume without loss of generality that the sequence is increasing.',
                decision=False,
                after=['Therefore the sequence converges.'],
            ),
            _end_demo(
                start='Exercise 1.2.1: Sketch the slope field.',
                candidate='Exercise 1.2.2: Sketch the next slope field.',
                decision=False,
            ),
        ]

    def encode(
        self,
        start_node: models.NodeInput,
        context_before: list[models.NodeInput],
        candidate_node: models.NodeInput,
        context_after: list[models.NodeInput],
    ) -> dict[str, object]:
        """Passes typed anchor, candidate, and context unchanged to DSPy."""
        return {
            'start_node': start_node,
            'context_before': context_before,
            'candidate_node': candidate_node,
            'context_after': context_after,
        }

    def decode(self, prediction: dspy.Prediction, **inputs: object) -> bool:
        """Returns a strictly boolean end decision."""
        return module.require_bool(
            prediction.is_pedagogical_end, 'is_pedagogical_end'
        )


def _target_context(
    nodes: list[models.SourceNode],
    position: int,
    before_budget: int,
    after_budget: int,
) -> tuple[list[models.NodeInput], models.NodeInput, list[models.NodeInput]]:
    before, target, after = context_window.select_target_context(
        nodes,
        position,
        before_budget=before_budget,
        after_budget=after_budget,
    )
    return (
        [
            context_window.node_input(node, index)
            for index, node in enumerate(before)
        ],
        context_window.node_input(target),
        [
            context_window.node_input(node, index)
            for index, node in enumerate(after)
        ],
    )


async def find_spans(
    nodes: list[models.SourceNode],
    start_router: PedagogicalStartRouter,
    end_router: PedagogicalEndRouter,
    start_before_budget: int,
    start_after_budget: int,
    end_before_budget: int,
    end_after_budget: int,
) -> list[list[int]]:
    """Finds contiguous pedagogical units with start and end routers."""
    spans: list[list[int]] = []
    cursor = 0
    while cursor < len(nodes):
        context_before, target_node, context_after = _target_context(
            nodes, cursor, start_before_budget, start_after_budget
        )
        is_start = module.require_bool(
            await start_router.aforward(
                context_before=context_before,
                target_node=target_node,
                context_after=context_after,
            ),
            'is_pedagogical_start',
        )
        if not is_start:
            cursor += 1
            continue

        block_start = cursor
        candidate = block_start
        start_node = context_window.node_input(
            context_window.project_nodes([nodes[block_start]])[0]
        )
        while candidate < len(nodes):
            context_before, candidate_node, context_after = _target_context(
                nodes, candidate, end_before_budget, end_after_budget
            )
            is_end = module.require_bool(
                await end_router.aforward(
                    start_node=start_node,
                    context_before=context_before,
                    candidate_node=candidate_node,
                    context_after=context_after,
                ),
                'is_pedagogical_end',
            )
            if is_end:
                spans.append(list(range(block_start, candidate + 1)))
                cursor = candidate + 1
                break
            candidate += 1
        else:
            raise ValueError(
                f'pedagogical unit starting at cursor {block_start} '
                'has no end before the node stream ends'
            )

    logger.info(
        'pedagogical component finder: %d nodes -> %d span(s)',
        len(nodes),
        len(spans),
    )
    return spans


class PedagogicalComponentFinderNode:
    """Graph node that finds units over nodes not claimed by instructions."""

    def __init__(
        self,
        start_router: PedagogicalStartRouter,
        end_router: PedagogicalEndRouter,
    ) -> None:
        self.start_router = start_router
        self.end_router = end_router

    async def run(self, state: state.State) -> dict:
        """Finds unit spans over nodes not claimed by an instruction."""
        nodes = state.get('nodes', [])
        excluded_positions = {
            member
            for instruction in state.get('instructions', [])
            for member in instruction.member_positions
        }
        eligible_positions = [
            position
            for position in range(len(nodes))
            if position not in excluded_positions
        ]
        eligible = [nodes[position] for position in eligible_positions]
        finder_config = config.get_settings().stages.finders.pedagogical
        local_spans = await find_spans(
            eligible,
            start_router=self.start_router,
            end_router=self.end_router,
            start_before_budget=finder_config.start_before_budget,
            start_after_budget=finder_config.start_after_budget,
            end_before_budget=finder_config.end_before_budget,
            end_after_budget=finder_config.end_after_budget,
        )
        spans = [
            [eligible_positions[position] for position in span]
            for span in local_spans
        ]
        return {'spans': spans}
