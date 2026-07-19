import dspy
from pydantic import BaseModel, Field


class DSPyModel(BaseModel):
    type: str | None = Field(
        default=None,
        description="The type of the node"
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

    NODE TYPES:
    - Paragraph: Standard prose text. Inline math remains in Paragraph.
    - Math: Standalone display math block (e.g. `$$ ... $$`).
    - Code: Fenced code block.
    - Table: Markdown table body only (grid rows). Do not put standalone caption or title
      lines inside Table — those belong in Caption when they appear as separate blocks.
    - Caption: Figure captions, table titles, notes, or labels when shown as separate prose
      blocks from the picture placeholder or table grid. Include identifiers (e.g. "Figure 3.2",
      "Table 4.") and all descriptive text for that asset. Emit one Caption per distinct block.
      Pairing to figures/tables is by document order (nearest preceding Image or Table); consumers
      handle caption-above-figure ordering when needed.
    - Image: Indexed placeholder only: `![N]()` where `N` matches the OCR picture index for that
      slot. Do not put caption prose in Image — use Caption node(s) for any labels or explanatory text.
      Never embed file paths in Image content; paths live on the node's `src` field after merging.
    - List: Bullet/numbered informational list (steps, features, recall items) that is NOT student exercises/problems.
      Emit one List node per list; item-level splitting happens later.
    - Admonition: Boxed/callout instructional content such as Examples,
      Notes, Tips, Warnings, Theorems, or similar textbook sidebars.
      Preserve any label/header and identifier when present
      (e.g. `Example 4.2`, `Theorem 7`, `Note`).
    - Instruction: Shared lead instruction that governs a group of exercises (e.g. `1-20 Find ...`).
      Emit exactly one Instruction node immediately before the governed Activity nodes.
      Include only the lead text (no individual exercise content).
    - Activity: A student exercise/problem/question to solve, not a worked example/demonstration.
      Typically appears near end-of-section/chapter under headings like Exercises/Problems/Practice/Review.
      If supporting material (table/image/graph/scenario) is attached to an exercise, keep it inside that Activity node.
    - Header: A heading/title for a section/chapter/exercise/problem/etc.
      Emit exactly one Header node for each heading; do not split headings into multiple nodes.

    ACTIVITY GROUPING:
    - Always emit one Activity node per numbered/bulleted exercise; never bundle multiple exercises together.
    - Two common forms:
      1) Single self-contained exercise, possibly with subparts `(a)(b)(c)` or `(i)(ii)(iii)`.
         Treat stem + all subparts as one Activity node.
      2) Shared lead instruction + list of exercises.
         Emit one Instruction node followed by one Activity node per item.
         Each Activity node contains only its own item content.
         Do not repeat the lead instruction inside Activity nodes.
         If lead text appears only in `previous_node_context`, still emit an Instruction node with that lead text.
    - Do not classify exercise lists as List nodes.
    - Do not emit standalone section/chapter Heading nodes as separate nodes.
      Keep heading text only when it is part of a structured node's own content
      (e.g. figure/table/list/admonition labels).
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
