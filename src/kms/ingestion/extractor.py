"""Structural node extraction — parses OCR markdown into a flat ordered AST.

Each textbook page's markdown is segmented into top-level block nodes
(paragraph, math, list, header, table, image, caption, code) by a DSPy
ChainOfThought module. The result is a flat list of structural nodes per page —
purely structural, no math-semantic typing (that lives in the entity layer).
"""

import asyncio
import logging

import dspy
from langgraph.types import Send
from pydantic import BaseModel, Field

from kms.core import llm, logs, models, state

logger = logging.getLogger(__name__)


_TYPE_MAP: dict[str, type[models.ASTNode]] = {
    'paragraph': models.ParagraphNode,
    'math': models.MathNode,
    'code': models.CodeNode,
    'list': models.ListNode,
    'table': models.TableNode,
    'image': models.ImageNode,
    'caption': models.CaptionNode,
    'header': models.HeaderNode,
}


def _node_for(node_type: str, content: str | None) -> models.ASTNode:
    """Create the right ASTNode subclass for one extracted block.

    Args:
        node_type: The type string the LLM emitted.
        content: The block's markdown.

    Returns:
        The node, falling back to a paragraph for an unknown type.
    """
    node_class = _TYPE_MAP.get(node_type, models.ParagraphNode)
    return node_class(content=content)


class DSPyModel(BaseModel):
    """A single extracted block node from the LLM: its type and content."""

    type: str = Field(
        description='The block type: paragraph, math, code, list, table, image, caption, or header.'
    )
    content: str | None = Field(
        default=None, description='The content of the node'
    )


class Signature(dspy.Signature):
    r"""
    Parse the markdown of one textbook page into a flat list of top-level
    structural nodes, in document order.

    LATEX FORMAT: All mathematical notation must use LaTeX format. Use single
    dollar signs `$ $` for inline math and double dollar signs `$$ $$` for
    block/display math.

    This extractor is purely STRUCTURAL and domain-agnostic: emit only general
    document structure. Do NOT try to identify math-semantic units (definitions,
    theorems, problems, exercises) or attach any subject-specific meaning. Your
    job is faithful block segmentation of the markdown, nothing more.

    EXTRACTION RULES:
    - Extract nodes from the given markdown, in document order.
    - One node per top-level markdown block, as the block appears. A node is the
      outermost structural unit (a paragraph, a display-math block, a list, a
      table, a heading, …); do not break a block's sub-parts into separate
      nodes, and do not merge distinct blocks into one. Segment on structure
      (block boundaries) only — never on meaning: do NOT split a block because
      of what it says (e.g. a paragraph that runs into "Proof." or "Solution."
      stays one node).
    - If content starts or ends abruptly at the boundary of the given markdown,
      extract it as-is — do not try to complete or trim it.

    NODE TYPES (emit `type` as exactly one of these values):
    - paragraph: Standard prose text. Inline math remains in the paragraph.
      Callout/sidebar prose (Notes, Tips, Warnings, worked Examples, Theorems,
      exercises) with no better fit goes here. When in doubt, a block of text is
      a paragraph.
    - math: Standalone display math block (e.g. `$$ ... $$`).
    - code: Fenced code block.
    - list: Bullet/numbered list (steps, features, recall items, or a run of
      exercises). Emit the whole list as a single list node — do not split it
      into per-item nodes.
    - table: Markdown table body only (grid rows). Do not put standalone caption
      or title lines inside table — those belong in caption when they appear as
      separate blocks.
    - image: Indexed placeholder only: `![N]()` where `N` matches the OCR
      picture index for that slot. Do not put caption prose in image — use
      caption node(s) for any labels or explanatory text. Never embed file paths
      in image content; paths live on the node's `src` field after merging.
    - caption: Figure captions, table titles, notes, or labels when shown as
      separate prose blocks from the picture placeholder or table grid. Include
      identifiers (e.g. "Figure 3.2", "Table 4.") and all descriptive text for
      that asset. Emit one caption per distinct block.
    - header: A heading/title for a section/chapter/exercise set/etc. Emit
      exactly one header node per heading; do not split a heading into multiple
      nodes. A short label that opens a labelled block (e.g. "Example 6.7",
      "Theorem 2.1", "Exercise 12") is a header."""

    segment_markdown: str = dspy.InputField(
        description='The raw markdown content of one textbook segment. Emit nodes for this content only.'
    )

    nodes: list[DSPyModel] = dspy.OutputField(
        description=(
            'Flat list of top-level nodes extracted from segment_markdown. Follow the class docstring for taxonomy and extraction rules.'
        )
    )


class Extractor(dspy.Module):
    """Parses one page's OCR markdown into structural block nodes.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.extractor = dspy.ChainOfThought(Signature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, segment_markdown: str) -> list[DSPyModel]:
        """Parse one page.

        Args:
            segment_markdown: The page's markdown.

        Returns:
            The page's top-level structural nodes, in document order.
        """
        result = await self.extractor.acall(segment_markdown=segment_markdown)
        nodes = list(result.nodes or [])
        logger.debug(
            'extract: %d chars -> %d node(s) | %s',
            len(segment_markdown or ''),
            len(nodes),
            logs.counts([str(node.type or '') for node in nodes]),
        )
        return nodes

    def forward(self, segment_markdown: str) -> list[DSPyModel]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(segment_markdown))


# --- LangGraph node: parse each segment's markdown into AST nodes ---


class ExtractorNode:
    """Fans out per-segment workers and collects the extracted AST.

    Args:
        module: The extractor module. Created fresh if None.
    """

    def __init__(self, module: Extractor | None = None) -> None:
        self.module = module or Extractor()

    def dispatch(self, state: state.State) -> list[Send] | str:
        """Fan out one worker per segment that has OCR'd content.

        Each segment is parsed in isolation — no neighbour context. Passing a
        segment's neighbours as context made the LLM bleed their content into
        this segment's node list (a measured ~25% duplicate-entity inflation on
        dense pages). Cross-segment continuations are healed downstream by the
        seam merger, so the extractor needs only its own page.

        Args:
            state: The pipeline state, holding the segment backbone.

        Returns:
            One Send per segment with content, or the collect step's name when
            none qualify.
        """
        segments = state.get('segments', [])
        sends = [
            Send('extractor_worker', {'segment': segment})
            for segment in segments
            if segment.content
        ]
        return sends or 'extractor_collect'

    async def worker(self, state: dict) -> dict:
        """Parse one segment's markdown into a flat list of AST nodes.

        Args:
            state: The worker payload, holding its ``segment``.

        Returns:
            The segment's ``extract_results`` entry.
        """
        segment: models.Segment = state['segment']
        extracted = await self.module.aforward(segment_markdown=segment.content)
        nodes = [_node_for(node.type, node.content) for node in extracted]
        return {'extract_results': [(segment.index, nodes)]}

    def collect(self, state: state.State) -> dict:
        """Merge each segment's extracted nodes back into the backbone.

        Args:
            state: The pipeline state, holding the extraction results.

        Returns:
            The updated segment backbone.
        """
        results = state.get('extract_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'nodes'
        )
        logger.info(
            'extractor: %d page(s) -> %d node(s)',
            len(results),
            sum(len(nodes) for _, nodes in results),
        )
        return {'segments': segments}
