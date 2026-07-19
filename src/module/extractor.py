import dspy
from pydantic import BaseModel, Field
from langgraph.types import Send

from .state import State, Segment, ASTNode, NodeType


class DSPyModel(BaseModel):
    type: NodeType = Field(
        description="The block type of the node — must be one of the NodeType values."
    )
    content: str | None = Field(
        default=None,
        description="The content of the node"
    )


class Signature(dspy.Signature):
    r"""
    Parse raw markdown from a single textbook segment into a flat list of top-level Node objects.
    A dedicated seam-healing step handles cross-segment continuations after extraction.

    LATEX FORMAT:
    All mathematical notation must use LaTeX format. Use single dollar signs `$ $` for inline math and double dollar signs `$$ $$` for block/display math.

    EXTRACTION RULES:
    - Extract nodes from `current_node` only, in document order.
    - `previous_node_context` and `next_node_context` are read-only and only for resolving classification ambiguity.
    - A top-level node is the outermost structural unit; do not break sub-parts into separate nodes.
    - If content starts or ends abruptly at a segment boundary, extract it as-is.

    NODE TYPES (emit `type` as exactly one of these values):
    - paragraph: Standard prose text. Inline math remains in the paragraph. Callout/sidebar
      prose (Notes, Tips, Warnings, worked Examples, Theorems) with no better fit goes here.
    - math: Standalone display math block (e.g. `$$ ... $$`).
    - code: Fenced code block.
    - list: Bullet/numbered informational list (steps, features, recall items) that is NOT
      student exercises/problems. Emit one list node per list; item-level splitting happens later.
    - table: Markdown table body only (grid rows). Do not put standalone caption or title
      lines inside table — those belong in caption when they appear as separate blocks.
    - image: Indexed placeholder only: `![N]()` where `N` matches the OCR picture index for that
      slot. Do not put caption prose in image — use caption node(s) for any labels or explanatory text.
      Never embed file paths in image content; paths live on the node's `src` field after merging.
    - caption: Figure captions, table titles, notes, or labels when shown as separate prose
      blocks from the picture placeholder or table grid. Include identifiers (e.g. "Figure 3.2",
      "Table 4.") and all descriptive text for that asset. Emit one caption per distinct block.
      Pairing to figures/tables is by document order (nearest preceding image or table); consumers
      handle caption-above-figure ordering when needed.
    - header: A heading/title for a section/chapter/exercise set/etc. Emit exactly one header node
      per heading; do not split a heading into multiple nodes.
    - instruction: Shared lead instruction that governs a group of exercises (e.g. `1-20 Find ...`).
      Emit exactly one instruction node immediately before the governed exercise nodes.
      Include only the lead text (no individual exercise content). [math-book specific]
    - exercise: A student exercise/problem/question to solve, not a worked example/demonstration.
      Typically appears near end-of-section/chapter under headings like Exercises/Problems/Practice/Review.
      If supporting material (table/image/graph/scenario) is attached to an exercise, keep it inside
      that exercise node. [math-book specific]

    EXERCISE GROUPING:
    - Always emit one exercise node per numbered/bulleted exercise; never bundle multiple exercises together.
    - Two common forms:
      1) Single self-contained exercise, possibly with subparts `(a)(b)(c)` or `(i)(ii)(iii)`.
         Treat stem + all subparts as one exercise node.
      2) Shared lead instruction + list of exercises.
         Emit one instruction node followed by one exercise node per item.
         Each exercise node contains only its own item content.
         Do not repeat the lead instruction inside exercise nodes.
         If lead text appears only in `previous_node_context`, still emit an instruction node with that lead text.
    - Do not classify exercise lists as list nodes.
    """

    previous_node_context: str | None = dspy.InputField(
        description=(
            "The segment immediately before the current one. Read-only — use only to resolve "
            "classification ambiguity. Do not emit nodes for this content."
        )
    )

    current_node: str = dspy.InputField(
        description="The raw markdown content of the current textbook segment. Emit nodes for this content only."
    )

    next_node_context: str | None = dspy.InputField(
        description=(
            "The segment immediately after the current one. Read-only — use only to resolve "
            "classification ambiguity. Do not emit nodes for this content."
        )
    )

    nodes: list[DSPyModel] = dspy.OutputField(
        description=(
            "Flat list of top-level nodes extracted from current_node. Follow the class docstring for taxonomy and extraction rules."
        )
    )


class Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(Signature)

    async def aforward(
        self,
        current_node: str,
        previous_node_context: str | None = None,
        next_node_context: str | None = None,
    ):
        result = await self.extractor.acall(
            previous_node_context=previous_node_context,
            current_node=current_node,
            next_node_context=next_node_context,
        )
        return dspy.Prediction(nodes=result.nodes)


# --- LangGraph node: parse each segment's markdown into AST nodes ---

_module = Module()


def dispatch(state: State) -> list[Send] | str:
    """Fan out one worker per segment that has OCR'd content, with neighbour text as context."""
    segments = state.get("segments", [])
    sends = [
        Send("extractor_worker", {
            "segment": seg,
            "previous_content": segments[i - 1].content if i > 0 else None,
            "next_content": segments[i + 1].content if i < len(segments) - 1 else None,
        })
        for i, seg in enumerate(segments)
        if seg.content
    ]
    return sends or "extractor_collect"


async def extractor_worker(state: dict) -> dict:
    """Parse one segment's markdown into a flat list of AST nodes."""
    segment: Segment = state["segment"]
    prediction = await _module.aforward(
        current_node=segment.content,
        previous_node_context=state.get("previous_content"),
        next_node_context=state.get("next_content"),
    )
    nodes = [ASTNode(type=node.type, content=node.content) for node in prediction.nodes]
    return {"extract_results": [(segment.index, nodes)]}


def extractor_collect(state: State) -> dict:
    """Merge each segment's extracted AST nodes back into the ordered backbone."""
    nodes_by_index = dict(state.get("extract_results", []))
    for segment in state["segments"]:
        if segment.index in nodes_by_index:
            segment.nodes = nodes_by_index[segment.index]
    return {"segments": state["segments"]}
