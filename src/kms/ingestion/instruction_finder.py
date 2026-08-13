import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import content, logs, models, recording, state, walker

logger = logging.getLogger(__name__)
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowNode(BaseModel):
    position: int
    type: str
    content: str | None = None
    image_path: str | None = None


class Span(BaseModel):
    start: int = Field(
        description='First local position of the span (inclusive).'
    )
    end: int = Field(description='Last local position of the span (inclusive).')


class Signature(dspy.Signature):
    r"""
    Find the exercise LEAD-IN sections in a run of textbook nodes and return
    each as a span of node positions.

    A lead-in section is a shared instruction that introduces a run of OTHER
    exercises: a short directive with no reference number of its own
    ("In the following exercises, simplify each expression.", "For Exercises
    5-6, determine whether the set is a subspace.", "9-16 Sketch the polar
    curve."). It may name an explicit RANGE or none at all. A node that
    BEGINS WITH ITS OWN EXERCISE NUMBER ("1.15 Perform each multiplication.",
    "✓ 1.17 For a homomorphism …") is an EXERCISE, never a lead-in.

    A lead-in section is the lead-in directive together with its own
    continuation: lettered sub-parts ("a. … b. …") or any text that completes
    the directive itself belong IN the section. It STOPS before the run of
    NUMBERED exercises the directive governs — a node that opens with its own
    exercise number ("280.", "302.") is a separate block, never part of the
    section. A directive with no continuation is a span of just the lead-in
    node.

    Emit spans over the given nodes ONLY, using their `[position]` labels; a
    span is the inclusive [start, end] range it occupies. Return spans in
    document order. If there are no lead-ins in the window, return an empty
    list.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The look-ahead window's nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself. Emit spans '
            'over the `[position]` values only.'
        )
    )
    instruction_spans: list[Span] = dspy.OutputField(
        description='Every exercise lead-in section found in current_nodes, as position spans, in '
        'document order. Empty list if none.'
    )


class InstructionFinder(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.finder = dspy.Predict(Signature)
        self.finder.demos = [
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, find the gradient.',
                        ),
                        WindowNode(
                            position=1,
                            type='paragraph',
                            content='280. Find the gradient of $f(x, y) = x^2 + y^2$.',
                        ),
                        WindowNode(
                            position=2,
                            type='paragraph',
                            content='281. Find the gradient of $f(x, y) = xy$.',
                        ),
                    ]
                ),
                instruction_spans=[Span(start=0, end=0)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content='In the following exercises, simplify each expression.',
                        ),
                        WindowNode(
                            position=1,
                            type='list',
                            content='979. $17a + 9a$',
                        ),
                        WindowNode(
                            position=2,
                            type='list',
                            content='980. $18z + 9z$',
                        ),
                    ]
                ),
                instruction_spans=[Span(start=0, end=0)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content='For the following exercises, find equations of:',
                        ),
                        WindowNode(
                            position=1,
                            type='paragraph',
                            content='a. the tangent plane and',
                        ),
                        WindowNode(
                            position=2,
                            type='paragraph',
                            content='b. the normal line to the given surface at the given point.',
                        ),
                        WindowNode(
                            position=3,
                            type='paragraph',
                            content='302. $z = 4x^2 + y^2$, point $P(2, 1, 8)$',
                        ),
                    ]
                ),
                instruction_spans=[Span(start=0, end=2)],
            ).with_inputs('current_nodes'),
            dspy.Example(
                current_nodes=content.labeled_content_parts(
                    [
                        WindowNode(
                            position=0,
                            type='paragraph',
                            content='282. Find the gradient of $f(x, y, z)$ at $P$ and the directional derivative in the direction of $\\mathbf{u}$.',
                        ),
                    ]
                ),
                instruction_spans=[],
            ).with_inputs('current_nodes'),
        ]
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, current_nodes: list[WindowNode]) -> list[Span]:
        result = await self.finder.acall(
            current_nodes=content.labeled_content_parts(current_nodes)
        )
        if self._recorder:
            self._recorder.record(
                'instruction_finder',
                {'current_nodes': current_nodes},
                result,
            )
        spans = list(result.instruction_spans or [])
        logger.debug(
            'find: %d nodes in, %d lead-in span(s) out | first node %r',
            len(current_nodes),
            len(spans),
            logs.elide(current_nodes[0].content if current_nodes else ''),
        )
        return spans

    def forward(self, current_nodes: list[WindowNode]) -> list[Span]:
        return asyncio.run(self.aforward(current_nodes))


def _normalize_spans(spans: list[Span], last_local: int) -> list[Span]:
    clamped: list[Span] = []
    for span in spans:
        start = min(max(span.start, 0), last_local)
        end = min(max(span.end, start), last_local)
        clamped.append(Span(start=start, end=end))
    clamped.sort(key=lambda span: (span.start, span.end))
    return clamped


async def find_instruction_spans(
    nodes: list[models.ASTNode],
    module: InstructionFinder,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[list[int]]:
    module = module
    spans_out: list[list[int]] = []
    cursor, node_count = 0, len(nodes)

    while cursor < node_count:
        size = budget
        while True:
            end = walker.window_from(nodes, cursor, size)
            window = nodes[cursor:end]
            last_local = len(window) - 1
            reached_doc_end = end == node_count

            spans = await module.aforward(
                [
                    WindowNode(
                        position=position,
                        type=node.type,
                        content=node.content,
                        image_path=node.image_path,
                    )
                    for position, node in enumerate(window)
                ]
            )
            clean = _normalize_spans(spans, last_local)

            if not clean:
                cursor = end
                break
            bounded = [span for span in clean if span.end < last_local]

            if reached_doc_end or size >= max_budget:
                if not reached_doc_end:
                    logger.warning(
                        'window hit the %d-token cap at cursor %d; banking %d '
                        'span(s) as-is (a span may be truncated)',
                        max_budget,
                        cursor,
                        len(clean),
                    )
                to_bank, advance = clean, end
            elif bounded:
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                logger.debug(
                    'grow: sole span reaches the window edge at cursor %d; budget %d -> %d',
                    cursor,
                    size,
                    size * 2,
                )
                size *= 2
                continue

            for span in to_bank:
                member_ids = [
                    window[position].id
                    for position in range(span.start, span.end + 1)
                    if window[position].id is not None
                ]
                if member_ids:
                    spans_out.append(member_ids)
            cursor = advance
            break

    logger.info(
        'instruction finder: %d nodes -> %d lead-in span(s)',
        node_count,
        len(spans_out),
    )
    return spans_out


class InstructionFinderNode:
    def __init__(self, module: InstructionFinder) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        spans = await find_instruction_spans(
            state.get('nodes', []), module=self.module
        )
        instructions = [
            models.Instruction(block=list(span), members=list(span))
            for span in spans
        ]
        return {'instructions': instructions}
