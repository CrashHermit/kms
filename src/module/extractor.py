import dspy
from langgraph.types import Send
from pydantic import BaseModel, Field

from .llm import text_lm
from .state import ASTNode, NodeType, Segment, State


class DSPyModel(BaseModel):
    type: NodeType = Field(
        description="The block type of the node — must be one of the NodeType values."
    )
    content: str | None = Field(default=None, description="The content of the node")


class Signature(dspy.Signature):
    r"""
    Parse the markdown of one textbook page into a flat list of top-level structural
    nodes, in document order.

    LATEX FORMAT:
    All mathematical notation must use LaTeX format. Use single dollar signs `$ $` for inline math and double dollar signs `$$ $$` for block/display math.

    This extractor is purely STRUCTURAL and domain-agnostic: emit only general document
    structure. Do NOT try to identify math-semantic units (definitions, theorems,
    problems, exercises) or attach any subject-specific meaning. Your job is faithful
    block segmentation of the markdown, nothing more.

    EXTRACTION RULES:
    - Extract nodes from the given markdown, in document order.
    - One node per top-level markdown block, as the block appears. A node is the
      outermost structural unit (a paragraph, a display-math block, a list, a table, a
      heading, …); do not break a block's sub-parts into separate nodes, and do not
      merge distinct blocks into one. Segment on structure (block boundaries) only —
      never on meaning: do NOT split a block because of what it says (e.g. a paragraph
      that runs into "Proof." or "Solution." stays one node).
    - If content starts or ends abruptly at the boundary of the given markdown, extract
      it as-is — do not try to complete or trim it.

    NODE TYPES (emit `type` as exactly one of these values):
    - paragraph: Standard prose text. Inline math remains in the paragraph. Callout/sidebar
      prose (Notes, Tips, Warnings, worked Examples, Theorems, exercises) with no better fit
      goes here. When in doubt, a block of text is a paragraph.
    - math: Standalone display math block (e.g. `$$ ... $$`).
    - code: Fenced code block.
    - list: Bullet/numbered list (steps, features, recall items, or a run of exercises).
      Emit the whole list as a single list node — do not split it into per-item nodes.
    - table: Markdown table body only (grid rows). Do not put standalone caption or title
      lines inside table — those belong in caption when they appear as separate blocks.
    - image: Indexed placeholder only: `![N]()` where `N` matches the OCR picture index for that
      slot. Do not put caption prose in image — use caption node(s) for any labels or explanatory text.
      Never embed file paths in image content; paths live on the node's `src` field after merging.
    - caption: Figure captions, table titles, notes, or labels when shown as separate prose
      blocks from the picture placeholder or table grid. Include identifiers (e.g. "Figure 3.2",
      "Table 4.") and all descriptive text for that asset. Emit one caption per distinct block.
    - header: A heading/title for a section/chapter/exercise set/etc. Emit exactly one header node
      per heading; do not split a heading into multiple nodes. A short label that opens a
      labelled block (e.g. "Example 6.7", "Theorem 2.1", "Exercise 12") is a header.
    """

    segment_markdown: str = dspy.InputField(
        description="The raw markdown content of one textbook segment. Emit nodes for this content only."
    )

    nodes: list[DSPyModel] = dspy.OutputField(
        description=(
            "Flat list of top-level nodes extracted from segment_markdown. Follow the class docstring for taxonomy and extraction rules."
        )
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.extractor = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, segment_markdown: str):
        result = await self.extractor.acall(segment_markdown=segment_markdown)
        return dspy.Prediction(nodes=result.nodes)


# --- LangGraph node: parse each segment's markdown into AST nodes ---


class ExtractorNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per segment that has OCR'd content.

        Each segment is parsed in isolation — no neighbour context. Passing a
        segment's neighbours as context made the LLM bleed their content into this
        segment's node list (a measured ~25% duplicate-entity inflation on dense
        pages). Cross-segment continuations are healed downstream by the seam merger,
        so the extractor needs only its own page."""
        segments = state.get("segments", [])
        sends = [Send("extractor_worker", {"segment": seg}) for seg in segments if seg.content]
        return sends or "extractor_collect"

    async def worker(self, state: dict) -> dict:
        """Parse one segment's markdown into a flat list of AST nodes."""
        segment: Segment = state["segment"]
        prediction = await self.module.aforward(segment_markdown=segment.content)
        nodes = [ASTNode(type=node.type, content=node.content) for node in prediction.nodes]
        return {"extract_results": [(segment.index, nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's extracted AST nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("extract_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
