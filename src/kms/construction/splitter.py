"""Splits packed exercise runs into one node per exercise."""

import logging

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import content, identity, models, module, state, walker

logger = logging.getLogger(__name__)


class SplitExercise(BaseModel):
    """One exercise piece: its reference number and verbatim content."""

    number: str = Field(
        description="The exercise's own reference number as written, e.g. '1.23'. EMPTY for a leading continuation fragment that belongs to a previous exercise."
    )
    content: str = Field(
        description="The piece's own text, copied verbatim, with its subparts, WITHOUT the leading number."
    )


class NodeSplit(BaseModel):
    """One packed node and the exercises it should be split into."""

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
    """Accumulated per-node split decisions keyed by node id."""

    splits: dict[int, list[SplitExercise]] = {}


class Splitter(module.Module):
    """Finds nodes packing multiple exercises and splits them."""

    signature = Signature
    record_name = 'splitter'

    def encode(
        self,
        current_nodes: list[walker.WindowNode],
        context_before: str | None = None,
    ) -> dict:
        """Builds the splitter-signature kwargs for one window."""
        return {
            'current_nodes': content.labeled_content_parts(current_nodes),
            'context_before': context_before or '',
        }

    def decode(self, prediction, **inputs) -> list[NodeSplit]:
        """Returns the split decisions from the prediction."""
        return module.as_list(prediction.splits)


async def _gather_decisions(
    nodes: list[models.Node], module: Splitter, budget: int
) -> Decision:
    """Walks the node stream in windows, collecting split decisions."""
    decision = Decision()
    cursor, node_count = 0, len(nodes)
    while cursor < node_count:
        end = walker.window_from(nodes, cursor, budget)
        window = nodes[cursor:end]
        last_local = len(window) - 1
        splits = await module.aforward(
            current_nodes=walker.node_views(window),
            context_before=walker.content_before(
                nodes,
                cursor,
                config.get_settings().stages.splitter.backward_context_budget,
            ),
        )
        seen_positions: set[int] = set()
        for split_result in splits:
            position = split_result.position
            if not 0 <= position <= last_local:
                raise ValueError(
                    f'invalid splitter position {position} for window '
                    f'of {len(window)} node(s) at cursor {cursor}'
                )
            if position in seen_positions:
                raise ValueError(
                    f'duplicate splitter position {position} at cursor {cursor}'
                )
            seen_positions.add(position)
            stream_position = cursor + position
            if stream_position >= len(nodes):
                raise ValueError(
                    f'splitter position {position} references a position '
                    f'outside node stream at cursor {cursor}'
                )
            if len(split_result.exercises) < 2:
                raise ValueError(
                    f'splitter position {position} returned fewer than two '
                    'exercise items'
                )
            if any(
                not exercise.content and not exercise.number
                for exercise in split_result.exercises
            ):
                raise ValueError(
                    f'splitter position {position} returned an empty '
                    'exercise item'
                )
            decision.splits[stream_position] = split_result.exercises
        cursor = end
    return decision


def _rebuild(nodes: list[models.Node], decision: Decision) -> list[models.Node]:
    """Rebuilds the node stream with split nodes expanded in place.

    Preserves UUIDs from original nodes.
    """
    out: list[models.Node] = []
    for position, node in enumerate(nodes):
        pieces = decision.splits.get(position)
        if pieces:
            for item in pieces:
                number = item.number or ''
                body = item.content or ''
                content = f'{number} {body}' if number else body
                out.append(
                    models.Node(
                        type=node.type,
                        content=content,
                        document_index=node.document_index,
                        uuid=node.uuid,
                        provenance=node.provenance,
                    )
                )
        else:
            out.append(node)
    return out


async def split_exercises(
    nodes: list[models.Node],
    module: Splitter,
    budget: int | None = None,
) -> list[models.Node]:
    """Splits packed exercises across the whole node stream.

    Args:
        nodes: The node stream, with ids assigned.
        module: The splitter module.
        budget: Look-ahead token budget per window.

    Returns:
        The rebuilt node stream with splits applied.
    """
    if not nodes:
        return nodes
    if budget is None:
        budget = config.get_settings().stages.splitter.lookahead_budget
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
    """Graph node that applies exercise splitting to the node stream."""

    def __init__(self, module: Splitter) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Splits the state's nodes and synchronizes document ownership."""
        source = state.get('source_key', '')
        nodes = await split_exercises(
            state.get('nodes', []), module=self.module
        )
        identity.assign_node_uuids(nodes, source)
        documents = state.get('documents', [])
        if documents:
            by_document: dict[int, list[models.Node]] = {
                document.index: [] for document in documents
            }
            for node in nodes:
                if node.document_index in by_document:
                    by_document[node.document_index].append(node)
            for document in documents:
                document.nodes = by_document[document.index]
            nodes = models.flatten_documents(documents)
        return {'documents': documents, 'nodes': nodes}
