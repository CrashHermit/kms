"""Finds the spans of pedagogical units in a run of nodes."""

import logging
from typing import cast

import dspy

from kms import config
from kms.construction.pedagogical_window_readiness import (
    PedagogicalWindowReadinessRouter,
)
from kms.core import (
    context_window,
    models,
    module,
    recording,
    state,
    walker,
)

logger = logging.getLogger(__name__)


class Signature(dspy.Signature):
    r"""
    Find the boundaries of every pedagogical unit in this node run. Return
    inclusive spans of one-based local ordinal indices. This is purely
    structural: do not decide whether a unit is a statement, procedure, or
    something to skip. A later role-typing pass makes that decision.

    FIND EVERY UNIT. Emit one span for each labelled definition, theorem,
    proposition, lemma, corollary, axiom, law, model, rule, or principle; each
    labelled or numbered worked example; each numbered exercise or problem;
    and each prescribed procedure whose ordered steps belong together.

    A unit starts at its own label or number. Include a separate label node such
    as "Example 6.7" or "Exercise 12" in that unit. Keep a stem and all
    lettered/subpart material together. Include any proof, solution, derivation,
    calculation, or computation output that resolves the unit. Distinct base
    numbers are always distinct units: never merge exercise 12 with exercise 13.

    A prescribed procedure's lead-in and consecutive numbered steps form one
    span. Step numbers order one unit; they do not create separate exercises.

    Do not emit spans for section headings, shared exercise instructions,
    ordinary narrative between units, or isolated formatting artifacts. The
    caller may already have removed shared instructions, but apply this rule
    regardless.

    OCR AND FRAGMENTED NODES:
    - Treat adjacent unnumbered nodes as one unit's continuation when the text
      clearly completes the preceding labelled or numbered unit.
    - Do not create a new unit for a dangling fragment such as a lone answer
      choice or continuation line.
    - Do not invent missing text or attach an unrelated fragment merely because
      it is nearby. If a fragment cannot be assigned confidently, leave it
      outside every span rather than corrupting a neighbouring unit.

    OUTPUT ONLY spans over the supplied nodes, using the explicit one-based
    `index` values. Do not use numbers from node text—such as exercise numbers,
    page numbers, years, or quantities—as span endpoints. If the window contains
    N nodes, every inclusive span must have both endpoints in the range 1
    through N; a one-node window can only produce span (1, 1) or no span.
    Never emit an endpoint beyond the final supplied node. Return spans in
    document order, with no overlap or duplicate ownership.
    Emit each exact span at most once. If uncertain, omit it rather than
    repeating it.
    Include a unit unfinished at the end of the window. Return an empty list
    when no unit is present.
    """

    current_nodes: list[models.NodeInput] = dspy.InputField(
        description=(
            'Ordered local node records with one-based index, node_type, and '
            'text. Use only index for span endpoints; text numbers are content.'
        )
    )
    spans: list[walker.Span] = dspy.OutputField(
        description=(
            'Every pedagogical unit as inclusive one-based spans in document '
            'order. Empty list if none.'
        )
    )


def _pedagogical_inputs(
    nodes: list[context_window.ContextNode],
) -> list[models.NodeInput]:
    """Projects context nodes as one-based structured model inputs."""
    return [
        context_window.node_input(node, local_index)
        for local_index, node in enumerate(nodes)
    ]


def _deduplicate_spans(
    spans: list[walker.Span],
) -> tuple[list[walker.Span], list[walker.Span]]:
    """Keeps first ownership declaration for each exact span."""
    unique: list[walker.Span] = []
    duplicates: list[walker.Span] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        key = (span.start, span.end)
        if key in seen:
            duplicates.append(span)
            continue
        seen.add(key)
        unique.append(span)
    return unique, duplicates


class PedagogicalComponentFinder(module.Module):
    """Finds pedagogical unit spans (statements, examples, exercises)."""

    signature = Signature
    record_name = 'pedagogical_component_finder'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [  # pyright: ignore[reportAttributeAccessIssue]
            dspy.Example(
                current_nodes=_pedagogical_inputs(
                    [
                        context_window.ContextNode(
                            position=0,
                            type='paragraph',
                            content="**Exercise 1.2.1:** Sketch the slope field for $y' = e^{x-y}$.",
                        ),
                        context_window.ContextNode(
                            position=1,
                            type='paragraph',
                            content="**Exercise 1.2.2:** Sketch the slope field for $y' = x^2$.",
                        ),
                        context_window.ContextNode(
                            position=2,
                            type='paragraph',
                            content="**Exercise 1.2.3:** Sketch the slope field for $y' = y^2$.",
                        ),
                    ]
                ),
                spans=[
                    walker.Span(start=1, end=1),
                    walker.Span(start=2, end=2),
                    walker.Span(start=3, end=3),
                ],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=_pedagogical_inputs(
                    [
                        context_window.ContextNode(
                            position=0,
                            type='paragraph',
                            content='**Exercise 1.2.7:** Let $\\{x_n\\}$ be a sequence.',
                        ),
                        context_window.ContextNode(
                            position=1,
                            type='list',
                            content='a) Show that $\\lim x_n = 0$ iff $\\lim |x_n| = 0$.',
                        ),
                        context_window.ContextNode(
                            position=2,
                            type='list',
                            content='b) Find an example where $\\{|x_n|\\}$ converges and $\\{x_n\\}$ diverges.',
                        ),
                    ]
                ),
                spans=[walker.Span(start=1, end=3)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=_pedagogical_inputs(
                    [
                        context_window.ContextNode(
                            position=0,
                            type='paragraph',
                            content='**Theorem 2.1.10.** Every bounded monotone sequence converges.',
                        ),
                        context_window.ContextNode(
                            position=1,
                            type='paragraph',
                            content='Proof. Assume without loss of generality that the sequence is increasing.',
                        ),
                    ]
                ),
                spans=[walker.Span(start=1, end=2)],
            ).with_inputs('current_nodes'),
        ]

    def encode(self, **inputs: object) -> dict[str, object]:
        """Projects modality-neutral context into structured node records."""
        current_nodes = cast(
            list[context_window.ContextNode], inputs['current_nodes']
        )
        return {'current_nodes': _pedagogical_inputs(current_nodes)}

    def decode(self, prediction, **inputs) -> list[walker.Span]:
        """Converts one-based model spans to validated zero-based spans."""
        raw_spans = module.as_list(prediction.spans)
        if any(not isinstance(span, walker.Span) for span in raw_spans):
            raise TypeError('spans must contain walker.Span values')
        window_nodes = cast(
            list[context_window.ContextNode], inputs['current_nodes']
        )
        window_size = len(window_nodes)
        converted: list[walker.Span] = []
        for index, span in enumerate(raw_spans):
            if not 1 <= span.start <= span.end <= window_size:
                raise ValueError(
                    f'invalid one-based span {index} '
                    f'({span.start}, {span.end}) for window of '
                    f'{window_size} node(s); converted values would be '
                    f'({span.start - 1}, {span.end - 1})'
                )
            converted.append(
                walker.Span(start=span.start - 1, end=span.end - 1)
            )
        unique, duplicates = _deduplicate_spans(converted)
        if duplicates:
            logger.warning(
                'finder removed %d duplicate span(s): %s',
                len(duplicates),
                [(span.start, span.end) for span in duplicates],
            )
        try:
            return walker.validate_spans(unique, window_size)
        except ValueError:
            logger.error(
                'finder span validation failed: raw_spans=%s '
                'converted_spans=%s window_nodes=%s',
                [span.model_dump() for span in raw_spans],
                [span.model_dump() for span in unique],
                [
                    node.model_dump()
                    for node in _pedagogical_inputs(window_nodes)
                ],
            )
            raise


async def find_spans(
    nodes: list[models.SourceNode],
    module: PedagogicalComponentFinder,
    readiness_module: PedagogicalWindowReadinessRouter | None = None,
    budget: int | None = None,
    max_budget: int | None = None,
) -> list[list[int]]:
    """Finds all pedagogical unit spans across the whole node stream.

    Args:
        nodes: The node stream, with ids assigned.
        module: The finder module.
        readiness_module: Window-readiness router that gates span extraction.
        budget: Initial look-ahead token budget per window.
        max_budget: Cap on window growth before banking as-is.

    Returns:
        A list of member-id lists, one per pedagogical unit.
    """
    if budget is None:
        budget = config.get_settings().stages.finders.lookahead_budget
    if max_budget is None:
        max_budget = config.get_settings().stages.finders.max_lookahead_budget
    if readiness_module is None:
        raise TypeError('readiness_module is required')
    spans = await walker.find_spans(
        nodes, module, readiness_module, budget, max_budget
    )
    logger.info(
        'pedagogical component finder: %d nodes -> %d span(s)',
        len(nodes),
        len(spans),
    )
    return spans


class PedagogicalComponentFinderNode:
    def __init__(
        self,
        module: PedagogicalComponentFinder,
        readiness_module: PedagogicalWindowReadinessRouter,
    ) -> None:
        self.module = module
        self.readiness_module = readiness_module

    async def run(self, state: state.State) -> dict:
        """Finds unit spans over nodes not claimed by an instruction.

        Args:
            state: Pipeline state with nodes and instructions.

        Returns:
            A ``spans`` update for the state.
        """
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
        local_spans = await find_spans(
            eligible,
            module=self.module,
            readiness_module=self.readiness_module,
        )
        spans = [
            [eligible_positions[position] for position in span]
            for span in local_spans
        ]
        return {'spans': spans}
