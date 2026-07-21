import dspy
from pydantic import BaseModel, Field
from langgraph.types import Send

from .state import State, Segment, ASTNode, NodeType
from .llm import text_lm


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
    - Extract nodes from the segment markdown, in document order.
    - A top-level node is the outermost structural unit; do not break sub-parts into separate nodes.
    - If content starts or ends abruptly at a segment boundary, extract it as-is.
    - A run-in proof or worked solution starts a NEW node: when a block opens with (or a
      paragraph runs into) a `Proof.` / `Solution.` marker that begins the justification
      of a preceding statement, split there. Never merge a theorem/proposition/lemma
      statement with its proof, or a worked example's statement with its solution, into a
      single node — the statement is one node, the proof/solution begins another.

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
    - instruction: Shared lead instruction that governs a group of problems (e.g. `1-20 Find ...`).
      Emit exactly one instruction node immediately before the governed problem nodes.
      Include only the lead text (no individual problem content). [math-book specific]
    - problem: A single student problem to solve — an exercise, question, or practice
      item. Typically appears near end-of-section/chapter under headings like
      Exercises/Problems/Practice/Review. If supporting material (table/image/graph/
      scenario) is attached to a problem, keep it inside that problem node. (Worked
      examples with a solution are stitched into problems by a later grouping stage,
      not here.) [math-book specific]

    PROBLEM GROUPING:
    - Emit one problem node per distinct problem. Subparts of the same problem are
      never split into separate nodes; distinct problems are never bundled together.
    - Subparts come in two forms, both kept together in a single problem node:
      1) A stem followed by unnumbered parts `(a)(b)(c)` or `(i)(ii)(iii)`: keep the
         stem and all its parts as one problem node.
      2) A repeated base number with letter suffixes — `12a`, `12b`, `12c` — is ONE
         problem, not three. Emit a single problem node for base number `12` whose
         body holds every part together, each kept as a labelled subpart (a, b, c);
         factor out the repeated base number so the node reads `12. a) ... b) ... c) ...`.
         Preserve each part's own content verbatim — only the redundant base number is
         factored out. Distinct base numbers (`12`, `13`, `14`) stay separate problems.
    - Shared lead instruction + list of problems:
         Emit one instruction node followed by one problem node per item.
         Each problem node contains only its own item content.
         Do not repeat the lead instruction inside problem nodes.
    - Do not classify problem lists as list nodes.
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
        and shared instruction leads are attached positionally by the instruction
        governor, so the extractor needs only its own page."""
        segments = state.get("segments", [])
        sends = [
            Send("extractor_worker", {"segment": seg})
            for seg in segments
            if seg.content
        ]
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
