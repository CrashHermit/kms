import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import content, models, recording, state, walker

logger = logging.getLogger(__name__)
LOOKAHEAD_BUDGET = 2000
BACKWARD_CONTEXT_BUDGET = 200


class WindowNode(BaseModel):
    position: int
    type: str
    content: str | None = None
    image_path: str | None = None


class SplitExercise(BaseModel):
    number: str = Field(
        description="The exercise's own reference number as written, e.g. '1.23'. EMPTY for a leading continuation fragment that belongs to a previous exercise."
    )
    content: str = Field(
        description="The piece's own text, copied verbatim, with its subparts, WITHOUT the leading number."
    )


class NodeSplit(BaseModel):
    position: int = Field(
        description='The window position of the node that packs the exercises.'
    )
    exercises: list[SplitExercise] = Field(
        description='The individual exercises it holds, in order (two or more).'
    )


class Signature(dspy.Signature):
    r"""
    Normalise a run of textbook nodes for the exercise layer.

    SPLITS — find any single node that packs TWO OR MORE numbered exercises into
    one block (usually a `list` node like "1.23 … 1.24 … 1.25 …"). Return that
    node's position and its exercises IN ORDER, each with its own `number`
    ("1.23") and its own `content` (that exercise's statement text, copied
    VERBATIM — same wording, same LaTeX, same math, do not paraphrase, reflow,
    or drop any subpart — keeping its subparts (a)(b)(c) together and keeping
    any incidental markers like a leading "✓", but WITHOUT the reference
    number).

    PRESERVE A LEADING FRAGMENT: if the node BEGINS with text that belongs to a
    PREVIOUS exercise (a continuation the layout left at the top of this node —
    e.g. trailing subparts "(d) … (e) …" before the first numbered exercise
    here), return it as the FIRST item with an EMPTY `number` and that fragment
    as its verbatim `content`, so nothing is lost.

    Every character of the node must land in exactly one item, in order. A node
    holding only ONE exercise is NOT a split — leave it out. Worked examples,
    definitions, theorems, prose, and headers are never splits.

    Use the given `position` values, over the given nodes ONLY. The list may be
    empty.
    """

    current_nodes: content.ContentParts = dspy.InputField(
        description=(
            "The look-ahead window's nodes, in order. Each text node is a "
            'line `[position] (type): content`; each image node is a line '
            '`[position] (image):` followed by the image itself.'
        )
    )
    context_before: str | None = dspy.InputField(
        default=None,
        description=(
            'Optional text immediately before the window, in document '
            'order. CONTEXT ONLY — use it to place the exercises; never '
            'split or copy text from it.'
        ),
    )
    splits: list[NodeSplit] = dspy.OutputField(
        description='Nodes that pack two or more exercises, each split into its individual exercises.'
    )


class Decision(BaseModel):
    splits: dict[int, list[SplitExercise]] = {}


class Splitter(dspy.Module):
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.splitter = dspy.Predict(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self, current_nodes: list[WindowNode], context_before: str | None = None
    ) -> list[NodeSplit]:
        result = await self.splitter.acall(
            current_nodes=content.labeled_content_parts(current_nodes),
            context_before=context_before or '',
        )
        if self._recorder:
            self._recorder.record(
                'splitter', {'current_nodes': current_nodes}, result
            )
        splits = list(result.splits or [])
        logger.debug(
            'split: %d nodes in, %d split(s) out%s',
            len(current_nodes),
            len(splits),
            ''.join(
                f' | position {split.position} -> {len(split.exercises)} piece(s)'
                for split in splits
            ),
        )
        return splits

    def forward(
        self, current_nodes: list[WindowNode], context_before: str | None = None
    ) -> list[NodeSplit]:
        return asyncio.run(
            self.aforward(current_nodes, context_before=context_before)
        )


async def _gather_decisions(
    nodes: list[models.ASTNode], module: Splitter, budget: int
) -> Decision:
    decision = Decision()
    cursor, node_count = 0, len(nodes)
    while cursor < node_count:
        end = walker.window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        splits = await module.aforward(
            [
                WindowNode(
                    position=position,
                    type=node.type,
                    content=node.content,
                    image_path=node.image_path,
                )
                for position, node in enumerate(window)
            ],
            context_before=walker.content_before(
                nodes, cursor, BACKWARD_CONTEXT_BUDGET
            ),
        )
        for split_result in splits:
            clamped = min(max(split_result.position, 0), last_local)
            node_id = window[clamped].id
            items = [
                exercise
                for exercise in split_result.exercises
                if (exercise.content or '').strip()
                or (exercise.number or '').strip()
            ]
            if node_id is not None and len(items) >= 2:
                decision.splits[node_id] = items
        cursor = end
    return decision


def _rebuild(
    nodes: list[models.ASTNode], decision: Decision
) -> list[models.ASTNode]:
    out: list[models.ASTNode] = []
    for node in nodes:
        pieces = decision.splits.get(node.id)
        if pieces:
            for item in pieces:
                number = (item.number or '').strip()
                body = (item.content or '').strip()
                content = f'{number} {body}'.strip() if number else body
                out.append(
                    models.ASTNode(
                        type=node.type,
                        content=content,
                        segment_index=node.segment_index,
                    )
                )
        else:
            out.append(node)
    for i, node in enumerate(out):
        node.id = i
    return out


async def split_exercises(
    nodes: list[models.ASTNode],
    module: Splitter,
    budget: int = LOOKAHEAD_BUDGET,
) -> list[models.ASTNode]:
    module = module
    if not nodes:
        return nodes
    decision = await _gather_decisions(nodes, module, budget)
    rebuilt = _rebuild(nodes, decision)
    logger.info(
        'splitter: %d node(s) -> %d (%d packed node(s) split)',
        len(nodes),
        len(rebuilt),
        len(decision.splits),
    )
    return rebuilt


class SplitterNode:
    def __init__(self, module: Splitter) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        nodes = await split_exercises(
            state.get('nodes', []), module=self.module
        )
        return {'nodes': nodes}
