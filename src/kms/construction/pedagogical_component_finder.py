"""Finds the spans of pedagogical units in a run of nodes."""

import logging

import dspy

from kms import config
from kms.core import content, models, module, recording, state, walker

logger = logging.getLogger(__name__)


class Signature(dspy.Signature):
    r"""
    Find the boundaries of every pedagogical unit in this node run. Return
    inclusive spans of local `[position]` values. This is purely structural:
    do not decide whether a unit is a statement, procedure, or something to
    skip. A later role-typing pass makes that decision.

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

    OUTPUT ONLY spans over the supplied nodes, using local positions. Return
    spans in document order, with no overlap and no duplicate ownership. Do not
    skip a clearly labelled or numbered unit. Include a unit that is unfinished
    at the end of the window. Return an empty list when no unit is present.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The look-ahead window's nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself. Emit spans '
            'over the `[position]` values only.'
        )
    )
    spans: list[walker.Span] = dspy.OutputField(
        description='Every pedagogical unit found in current_nodes, as position spans, in '
        'document order — declarative statements, worked examples, exercises, '
        'and prescribed procedures alike. Boundaries only — do NOT classify '
        'them. Empty list if none.'
    )


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
        self.predictor.demos = [
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content="**Exercise 1.2.1:** Sketch the slope field for $y' = e^{x-y}$.",
                        ),
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            content="**Exercise 1.2.2:** Sketch the slope field for $y' = x^2$.",
                        ),
                        walker.WindowNode(
                            position=2,
                            type='paragraph',
                            content="**Exercise 1.2.3:** Sketch the slope field for $y' = y^2$.",
                        ),
                    ]
                ),
                spans=[
                    walker.Span(start=0, end=0),
                    walker.Span(start=1, end=1),
                    walker.Span(start=2, end=2),
                ],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='**Exercise 1.2.7:** Let $\\{x_n\\}$ be a sequence.',
                        ),
                        walker.WindowNode(
                            position=1,
                            type='list',
                            content='a) Show that $\\lim x_n = 0$ iff $\\lim |x_n| = 0$.',
                        ),
                        walker.WindowNode(
                            position=2,
                            type='list',
                            content='b) Find an example where $\\{|x_n|\\}$ converges and $\\{x_n\\}$ diverges.',
                        ),
                    ]
                ),
                spans=[walker.Span(start=0, end=2)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        walker.WindowNode(
                            position=0,
                            type='paragraph',
                            content='**Theorem 2.1.10.** Every bounded monotone sequence converges.',
                        ),
                        walker.WindowNode(
                            position=1,
                            type='paragraph',
                            content='Proof. Assume without loss of generality that the sequence is increasing.',
                        ),
                    ]
                ),
                spans=[walker.Span(start=0, end=1)],
            ).with_inputs('current_nodes'),
        ]

    def encode(self, current_nodes: list[walker.WindowNode]) -> dict:
        """Builds the finder-signature kwargs for one window."""
        return {'current_nodes': content.labeled_content_parts(current_nodes)}

    def decode(self, prediction, **inputs) -> list[walker.Span]:
        """Returns the unit spans from the prediction."""
        return module.as_list(prediction.spans)


async def find_spans(
    nodes: list[models.Node],
    module: PedagogicalComponentFinder,
    budget: int | None = None,
    max_budget: int | None = None,
) -> list[list[int]]:
    """Finds all pedagogical unit spans across the whole node stream.

    Args:
        nodes: The node stream, with ids assigned.
        module: The finder module.
        budget: Initial look-ahead token budget per window.
        max_budget: Cap on window growth before banking as-is.

    Returns:
        A list of member-id lists, one per pedagogical unit.
    """
    if budget is None:
        budget = config.get_settings().stages.finders.lookahead_budget
    if max_budget is None:
        max_budget = config.get_settings().stages.finders.max_lookahead_budget
    spans = await walker.find_spans(nodes, module, budget, max_budget)
    logger.info(
        'pedagogical component finder: %d nodes -> %d span(s)',
        len(nodes),
        len(spans),
    )
    return spans


class PedagogicalComponentFinderNode:
    """Graph node that finds unit spans outside instruction members."""

    def __init__(self, module: PedagogicalComponentFinder) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Finds unit spans over nodes not claimed by an instruction.

        Args:
            state: Pipeline state with nodes and instructions.

        Returns:
            A ``spans`` update for the state.
        """
        nodes = state.get('nodes', [])
        excluded = {
            member
            for instruction in state.get('instructions', [])
            for member in instruction.members
        }
        eligible = [node for node in nodes if node.id not in excluded]
        spans = await find_spans(eligible, module=self.module)
        return {'spans': spans}
