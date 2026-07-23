import dspy
from langgraph.types import Send
from pydantic import BaseModel

from .llm import text_lm
from .state import ASTNode, Segment, State, flatten_segments, merge_results_into_segments


class SeamNodeDTO(BaseModel):
    """Lightweight DSPy boundary model representing a node's content and type."""

    content: str | None = None
    types: list[str] = []


class Signature(dspy.Signature):
    """
    You are an expert technical editor. Two adjacent runs of document blocks share a seam
    — the boundary where one run ends and the next begins. Sometimes a single block
    (paragraph, sentence, equation, list item, caption, etc.) is split across that
    boundary, producing an incomplete tail in the top run and an incomplete head in the
    bottom run.

    Your job: decide whether the tail node of the top run and the head node of the bottom
    run are two halves of the same interrupted block. If they are, merge them into one
    coherent node. If they are not — they are already complete, independent nodes that
    merely sit next to each other at the boundary — return None.

    Judge this purely on structure: does the tail read as cut off mid-block and the head
    as its continuation? Do not reason about the subject matter or reassemble blocks that
    are each already complete.

    Use the context nodes (the neighbour just inside each run) only to inform your
    judgment — never include their content in the merged output.

    LATEX FORMAT:
    All mathematical notation must use LaTeX format. Use single dollar signs `$ $`
    for inline math and double dollar signs `$$ $$` for block/display math.
    Preserve existing delimiters and math content exactly.
    """

    top_node_context: SeamNodeDTO | None = dspy.InputField(
        description="The node immediately before the tail of the top element run. Read-only context — do not include its content in the output."
    )
    top_bottom_edge_node: SeamNodeDTO = dspy.InputField(
        description="The tail node of the top element run — the candidate for merging."
    )
    bottom_top_edge_node: SeamNodeDTO = dspy.InputField(
        description="The head node of the bottom element run — the other candidate for merging."
    )
    bottom_node_context: SeamNodeDTO | None = dspy.InputField(
        description="The node immediately after the head of the bottom element run. Read-only context — do not include its content in the output."
    )

    node: SeamNodeDTO | None = dspy.OutputField(
        description="The merged result. If the two edge nodes are split halves of the same node, return a single merged node combining their content. If they are already complete independent nodes, return None."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.merger = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ) -> SeamNodeDTO | None:
        result = await self.merger.acall(
            top_node_context=top_node_context,
            top_bottom_edge_node=top_bottom_edge_node,
            bottom_top_edge_node=bottom_top_edge_node,
            bottom_node_context=bottom_node_context,
        )
        return result.node


# --- LangGraph node: stitch nodes split across segment boundaries ---
#
# A worker touches two adjacent segments (top tail + bottom head), so adjacent
# pairs cannot run at once without racing on the shared segment. We run two passes:
# the even pass handles pairs whose top index is even (0-1, 2-3, ...), the odd pass
# handles the rest (1-2, 3-4, ...). Within a pass no two pairs share a segment, so
# they fan out safely; the passes run sequentially (even -> collect -> odd -> collect),
# and each pass writes its own reducer channel to avoid cross-pass contamination.


def _to_dto(node: ASTNode | None) -> SeamNodeDTO:
    if node is None:
        return SeamNodeDTO(content=None, types=[])
    return SeamNodeDTO(content=node.content, types=[node.type] if node.type else [])


def _pairs(segments: list[Segment], parity: int) -> list[tuple[Segment, Segment]]:
    return [
        (segments[i], segments[i + 1])
        for i in range(len(segments) - 1)
        if segments[i].index % 2 == parity and segments[i].nodes and segments[i + 1].nodes
    ]


async def _merge_pair(
    module: Module, top: Segment, bottom: Segment
) -> list[tuple[int, list[ASTNode]]]:
    """Decide whether the top's tail and the bottom's head are one split node; if so,
    fold the merged content into the tail and drop the head."""
    top_nodes = list(top.nodes)
    bottom_nodes = list(bottom.nodes)

    tail = top_nodes[-1]
    head = bottom_nodes[0]
    top_context = top_nodes[-2] if len(top_nodes) > 1 else None
    bottom_context = bottom_nodes[1] if len(bottom_nodes) > 1 else None

    merged = await module.aforward(
        top_bottom_edge_node=_to_dto(tail),
        bottom_top_edge_node=_to_dto(head),
        top_node_context=_to_dto(top_context),
        bottom_node_context=_to_dto(bottom_context),
    )
    if merged is not None and merged.content:
        tail.content = merged.content
        bottom_nodes = bottom_nodes[1:]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class SeamMergerNode:
    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    def dispatch_even(self, state: State) -> list[Send] | str:
        pairs = _pairs(state.get("segments", []), parity=0)
        sends = [Send("seam_even_worker", {"top": t, "bottom": b}) for t, b in pairs]
        return sends or "seam_even_collect"

    def dispatch_odd(self, state: State) -> list[Send] | str:
        pairs = _pairs(state.get("segments", []), parity=1)
        sends = [Send("seam_odd_worker", {"top": t, "bottom": b}) for t, b in pairs]
        return sends or "seam_odd_collect"

    async def even_worker(self, state: dict) -> dict:
        return {"seam_even_results": await _merge_pair(self.module, state["top"], state["bottom"])}

    async def odd_worker(self, state: dict) -> dict:
        return {"seam_odd_results": await _merge_pair(self.module, state["top"], state["bottom"])}

    def _collect(self, state: State, channel: str) -> dict:
        segments = merge_results_into_segments(state["segments"], state.get(channel, []), "nodes")
        return {"segments": segments}

    def even_collect(self, state: State) -> dict:
        return self._collect(state, "seam_even_results")

    def odd_collect(self, state: State) -> dict:
        """Drain the odd pass, then birth the flat global node list. The seam merger is
        the last stage that splits/merges nodes structurally, so page-splits are now
        healed and node identity is stable — flatten the per-page backbone into `nodes`,
        stamping each with its global id and originating seg_index. Every stage after
        this works on `nodes`, not on the per-segment nesting."""
        result = self._collect(state, "seam_odd_results")
        segments = result["segments"]
        return {"segments": segments, "nodes": flatten_segments(segments)}
