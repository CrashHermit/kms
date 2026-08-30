"""Splits packed exercise runs into one node per exercise."""

import logging

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import (
    context_window,
    identity,
    models,
    module,
    state,
    walker,
)

logger = logging.getLogger(__name__)


class SplitExercise(BaseModel):
    """One exercise piece with its number retained in verbatim content."""

    content: str = Field(
        description=(
            "The complete exercise text, including its leading number, "
            "copied verbatim."
        )
    )


class NodeSplit(BaseModel):
    """One packed node and the exercises it should be split into."""

    position: int = Field(
        description='The window position of the node that packs the exercises.'
    )
    exercises: list[SplitExercise] = Field(
        description='The individual exercises it holds, in order (two or more).'
    )



class ExerciseStripRouterSignature(dspy.Signature):
    r"""
    Classify only `target_node`. The before and after lists are context only.
    Return True only when the target node visibly packs TWO OR MORE distinct
    numbered exercises or problems that must become separate nodes.

    Return False for a single exercise, definition, theorem, proposition,
    proof, worked example, ordinary narrative, header, caption, shared
    instruction, or prescribed procedure with numbered steps. Numbered
    procedure steps are one procedure, not multiple exercises. When uncertain,
    return False.

    Return only a boolean. Answer only True or False.
    """

    context_before: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately before target_node; context only.',
    )
    target_node: models.NodeInput = dspy.InputField(
        description='The only one-based node record being classified as an exercise strip.',
    )
    context_after: list[models.NodeInput] = dspy.InputField(
        description='Ordered one-based node records immediately after target_node; context only.',
    )
    contains_multiple_exercises: bool = dspy.OutputField(
        description='True only when target_node contains two or more distinct exercises.',
    )




class ExerciseStripRouter(module.Module):
    """Routes nodes that pack multiple exercises to the splitter."""

    signature = ExerciseStripRouterSignature
    record_name = 'exercise_strip_router'

    def encode(
        self,
        context_before: list[models.NodeInput],
        target_node: models.NodeInput,
        context_after: list[models.NodeInput],
    ) -> dict[str, object]:
        """Passes text-only target and context to DSPy."""
        return {
            'context_before': context_before,
            'target_node': target_node,
            'context_after': context_after,
        }

    def decode(self, prediction, **inputs) -> bool:
        """Returns a strictly boolean routing decision."""
        return module.require_bool(
            prediction.contains_multiple_exercises,
            'contains_multiple_exercises',
        )


class Signature(dspy.Signature):
    r"""
    Normalise a run of textbook nodes for the exercise layer.

    SPLITS — find any single node that packs TWO OR MORE numbered exercises into
    one block (usually a `list` node like "1.23 … 1.24 … 1.25 …"). Return that
    node's position and its exercises IN ORDER. Each exercise's `content` must
    contain its own leading number ("1.23") and its complete statement text,
    copied VERBATIM — same wording, same LaTeX, same math, do not paraphrase,
    reflow, or drop any subpart — keeping its subparts (a)(b)(c) together and
    keeping incidental markers like a leading "✓".

    PRESERVE A LEADING FRAGMENT: if the node BEGINS with text that belongs to a
    PREVIOUS exercise (a continuation the layout left at the top of this node —
    e.g. trailing subparts "(d) … (e) …" before the first numbered exercise
    here), return it as the FIRST item's verbatim `content`, with no invented
    number, so nothing is lost.

    Every character of the node must land in exactly one item, in order. A node
    holding only ONE exercise is NOT a split — leave it out. Worked examples,
    definitions, theorems, prose, and headers are never splits.

    Use the given `position` values, over the given nodes ONLY. The list may be
    empty.
    """

    current_nodes: list[models.NodeInput] = dspy.InputField(
        description=(
            'Ordered look-ahead nodes. Each record has one-based index, '
            'node_type, and text. Return positions only from current_nodes; '
            'context-only nodes are not returnable.'
        )
    )
    context_before: list[models.NodeInput] = dspy.InputField(
        description=(
            'Ordered context-only nodes immediately before current_nodes. '
            'Their indexes are context-local and must not be returned.'
        )
    )
    splits: list[NodeSplit] = dspy.OutputField(
        description=(
            'Nodes that pack two or more exercises, each split into its '
            'individual exercises.'
        )
    )

class Splitter(module.Module):
    """Finds nodes packing multiple exercises and splits them."""

    signature = Signature
    record_name = 'splitter'

    def encode(
        self,
        current_nodes: list[models.NodeInput],
        context_before: list[models.NodeInput] | None = None,
    ) -> dict[str, object]:
        return {
            'current_nodes': current_nodes,
            'context_before': context_before or [],
        }

    def decode(self, prediction, **inputs) -> list[NodeSplit]:
        raw_splits = module.as_list(prediction.splits)
        if any(not isinstance(split, NodeSplit) for split in raw_splits):
            raise TypeError('splits must contain NodeSplit values')
        window_size = len(inputs['current_nodes'])
        splits: list[NodeSplit] = []
        for split in raw_splits:
            if not 1 <= split.position <= window_size:
                raise ValueError(
                    f'invalid one-based splitter position {split.position} '
                    f'for window of {window_size} node(s)'
                )
            if len(split.exercises) < 2:
                raise ValueError(
                    f'split at position {split.position} has fewer than '
                    'two exercise items'
                )
            for index, exercise in enumerate(split.exercises):
                if not exercise.content.strip():
                    raise ValueError(
                        f'split at position {split.position} has empty '
                        f'exercise item {index}'
                    )
            splits.append(split.model_copy(update={'position': split.position - 1}))
        module.require_positions(
            [split.position for split in splits],
            field_name='splits.position',
            upper_bound=window_size,
            ordered=True,
        )
        return splits

def _splitter_inputs(
    nodes: list[context_window.ContextNode],
) -> list[models.NodeInput]:
    """Projects context nodes into one-based structured records."""
    return [
        context_window.node_input(node, index)
        for index, node in enumerate(nodes)
    ]


class Decision(BaseModel):
    """Accumulated per-node split decisions keyed by node position."""

    splits: dict[int, list[SplitExercise]] = {}

def _rebuild(
    nodes: list[models.SourceNode], decision: Decision
) -> list[models.SourceNode]:
    """Rebuilds the node stream with split nodes expanded in place."""
    out: list[models.SourceNode] = []
    for position, node in enumerate(nodes):
        pieces = decision.splits.get(position)
        if pieces:
            if not node.uuid:
                raise ValueError(
                    f'cannot split node at position {position} without a uuid'
                )
            for child_index, item in enumerate(pieces):
                out.append(
                    models.SourceNode(
                        type=node.type,
                        content=item.content,
                        document_index=node.document_index,
                        uuid=identity.split_child_uuid(node.uuid, child_index),
                        assets=node.assets.copy(),
                        provenance=node.provenance,
                        governing_instruction_uuids=node.governing_instruction_uuids.copy(),
                    )
                )
        else:
            out.append(node)
    return out

async def _select_candidates(
    nodes: list[models.SourceNode],
    router: ExerciseStripRouter,
) -> set[int]:
    """Returns source positions routed to the exercise splitter."""
    selected: set[int] = set()
    context_budget = (
        config.get_settings().stages.finders.instruction_finder.context_budget
    )
    for position in range(len(nodes)):
        window = context_window.select_around(
            nodes,
            [position],
            backward_budget=0,
            forward_budget=context_budget,
            marker='exercise_target',
        )
        target_index = next(
            index for index, node in enumerate(window) if node.marker is not None
        )
        before = [
            context_window.node_input(node, index)
            for index, node in enumerate(window[:target_index])
        ]
        target = context_window.node_input(window[target_index])
        after = [
            context_window.node_input(node, index)
            for index, node in enumerate(window[target_index + 1 :])
        ]
        if await router.aforward(
            context_before=before,
            target_node=target,
            context_after=after,
        ):
            selected.add(position)
    return selected


async def _gather_decisions(
    nodes: list[models.SourceNode],
    module: Splitter,
    budget: int,
    selected_positions: set[int],
) -> Decision:
    """Walks the node stream in windows, collecting split decisions."""
    decision = Decision()
    cursor, node_count = 0, len(nodes)
    while cursor < node_count:
        end = walker.window_from(nodes, cursor, budget)
        candidate_positions = [
            position
            for position in range(cursor, end)
            if position in selected_positions
        ]
        if candidate_positions:
            window = nodes[cursor:end]
            current_nodes = [
                context_window.project_nodes(window)[position - cursor]
                for position in candidate_positions
            ]
            splits = await module.aforward(
                current_nodes=_splitter_inputs(current_nodes),
                context_before=_splitter_inputs(
                    context_window.project_nodes(
                        context_window.nodes_before(
                            nodes,
                            cursor,
                            config.get_settings().stages.splitter.backward_context_budget,
                        )
                    )
                ),
            )
            seen_positions: set[int] = set()
            for split_result in splits:
                position = split_result.position
                if not 0 <= position < len(candidate_positions):
                    raise ValueError(
                        f'invalid splitter position {position} for candidate '
                        f'window at cursor {cursor}'
                    )
                if position in seen_positions:
                    raise ValueError(
                        f'duplicate splitter position {position} at cursor {cursor}'
                    )
                seen_positions.add(position)
                stream_position = candidate_positions[position]
                if len(split_result.exercises) < 2:
                    raise ValueError(
                        f'splitter position {position} returned fewer than two '
                        'exercise items'
                    )
                if any(
                    not exercise.content.strip()
                    for exercise in split_result.exercises
                ):
                    raise ValueError(
                        f'splitter position {position} returned an empty '
                        'exercise item'
                    )
                decision.splits[stream_position] = split_result.exercises
        cursor = end
    return decision


async def split_exercises(
    nodes: list[models.SourceNode],
    module: Splitter,
    router: ExerciseStripRouter,
    budget: int | None = None,
) -> list[models.SourceNode]:
    """Routes and splits packed exercises across the whole node stream."""
    if not nodes:
        return nodes
    if budget is None:
        budget = config.get_settings().stages.splitter.lookahead_budget
    selected_positions = await _select_candidates(nodes, router)
    decision = await _gather_decisions(
        nodes,
        module,
        budget,
        selected_positions,
    )
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

    def __init__(
        self,
        module: Splitter,
        router: ExerciseStripRouter,
    ) -> None:
        self.module = module
        self.router = router

    async def run(self, state: state.State) -> dict:
        """Splits the state's nodes and synchronizes document ownership."""
        source = state['source'].key
        nodes = state.get('nodes', [])
        identity.assign_node_uuids(nodes, source)
        nodes = await split_exercises(
            nodes,
            module=self.module,
            router=self.router,
        )
        documents = state.get('documents', [])
        if documents:
            by_document: dict[int, list[models.SourceNode]] = {
                document.index: [] for document in documents
            }
            for node in nodes:
                if node.document_index in by_document:
                    by_document[node.document_index].append(node)
            for document in documents:
                document.nodes = by_document[document.index]
            nodes = models.flatten_documents(documents)
        return {'documents': documents, 'nodes': nodes}
