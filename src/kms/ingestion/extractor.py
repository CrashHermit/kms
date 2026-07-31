"""Structural node extraction — parses OCR markdown into a flat ordered AST.

Each textbook page's markdown is segmented into top-level block nodes
(paragraph, math, list, header, table, image, caption, code, bibliographic) by a
DSPy ChainOfThought module. The result is a flat list of structural nodes per
page — purely structural, no math-semantic typing (that lives in the entity
layer).

``furniture`` is the stage's one *discard* type. The apparatus around a
document — the running head, the folio, the colophon or licence line at the foot
of the page — is not part of what the document says, and since the OCR front end
appends each page's extracted footer back onto its markdown (so footnote
citations survive), that apparatus now reaches this stage. It is identified here
and dropped before the stage returns, so nothing downstream ever sees it: no
node, no id, no graph vertex. Identifying it *here* rather than in a later
finder is what keeps the seam merger correct — a colophon left at the foot of a
page would take the tail's place and prevent a genuinely split paragraph from
being healed, exactly as a footnote would.

Discarding is why it is a *type* and not a stage: the judgment is one the
extractor is already making (which blocks are there, and what each one is), and
a block that never becomes a node needs nothing else built for it. The cost is
that it is unrecoverable — a false positive deletes real content silently — so
the prompt is written to keep anything it is unsure about, and every dropped
block is logged at DEBUG.

``bibliographic`` and ``note`` are the two types whose test is not the block's
shape: a footnote and a reference-list entry are both prose paragraphs to look
at. What separates them from prose is what they are *for* — one names an
external work, the other hangs off a marker in the body — and between the two,
naming a work wins. ``bibliographic`` is decided here rather than in a later
semantic stage because a reference is a *block* — one node per work — and
blocking is this stage's job; nothing downstream can recover the entry
boundaries once several works are packed into one paragraph node.

Together with ``furniture`` these give the page's bottom edge a three-way split
with a positive test each, instead of forcing every footnote to be argued out of
the discard: furniture talks about the artifact and is dropped, a note talks
about the subject and is kept, a reference names a work. ``note`` nodes stay in
the ordinary stream and remain eligible for the semantic chain — a footnote that
defines a term is a definition wherever it happens to be printed.
"""

import asyncio
import logging

import dspy
from langgraph.types import Send
from pydantic import BaseModel, Field

from kms.core import llm, logs, models, recorder, state

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
    'bibliographic': models.BibliographicNode,
    'note': models.NoteNode,
}

# Block types the stage identifies in order to throw away. These never become
# nodes, so they have no ``models.ASTNode`` class and never reach the stream —
# see the module docstring on why the discard happens here.
_DISCARDED_TYPES = frozenset({'furniture'})


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
        description='The block type: paragraph, math, code, list, table, image, caption, header, bibliographic, note, or furniture.'
    )
    content: str | None = Field(default=None, description='The content of the node')


def _partition(
    blocks: list[DSPyModel],
) -> tuple[list[DSPyModel], list[DSPyModel]]:
    """Split one page's blocks into the ones kept and the ones discarded.

    A discarded block is page apparatus (``furniture``): identified so it can
    be thrown away here, before the stage returns, rather than travelling the
    pipeline as a node every later stage has to ignore.

    Args:
        blocks: The blocks the LLM emitted for one page, in document order.

    Returns:
        The ``(kept, discarded)`` blocks, each in document order.
    """
    kept: list[DSPyModel] = []
    discarded: list[DSPyModel] = []
    for block in blocks:
        target = discarded if (block.type or '').strip().lower() in _DISCARDED_TYPES else kept
        target.append(block)
    return kept, discarded


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
      stays one node). The single exception is a run of bibliographic
      references, which is split per cited work — see that type below.
    - If content starts or ends abruptly at the boundary of the given markdown,
      extract it as-is — do not try to complete or trim it, and NEVER leave it
      out. A page often opens or closes mid-block, so the first or last thing
      on it may be a bare number, a stray pair of values, an unpunctuated
      half-sentence, or a piece of a code listing. Such a fragment is content:
      fold it into the block it continues when that is clear (a numeric line
      directly above a code listing belongs INSIDE that code node), and give
      it its own node otherwise. It is rejoined to its other half downstream,
      but only if it survives this stage. A leading bare number is a fragment
      of this kind, not a page number.
      When it gets its own node, TYPE IT AS WHAT IT IS A PIECE OF, never by
      where it sits on the page: an unpunctuated half-sentence of prose is a
      paragraph, a run of code is code, a row of values is table. A fragment is
      NEVER a header — a heading is a short title that opens what follows, so
      text that starts lowercase, starts mid-sentence, or completes a sentence
      the previous page began cannot be one. The first line of a page is not a
      heading merely because it is first.
      Type it right or the repair never happens: the stage that rejoins the two
      halves downstream merges only nodes of the SAME type, so a fragment that
      survives with the wrong type is as lost as one deleted.

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
      "Theorem 2.1", "Exercise 12") is a header.
      Copy the heading line EXACTLY as the markdown has it, keeping its leading
      `#` markers and any bold or italic markup: the node for "## 1.5 Project"
      has content "## 1.5 Project", never "1.5 Project". The markers are what
      set the heading's level, and no later stage can recover a level that was
      stripped here.
      A RUN-IN HEADING — a label followed by body text on the SAME line, e.g.
      "**Steps** We recommend proceeding in the following order:" — is TWO
      nodes: the label as a header, and the text after it as its own node of
      whatever type that text is. Emitting only the label deletes the rest of
      the line. Never do that.
    - bibliographic: A reference to an external work — a published paper, book,
      chapter, report, or web resource. It cites a work rather than saying
      something: authors and a year with a title, and usually a venue,
      publisher, page range, DOI, or URL. It appears either as an entry in a
      reference list ("References", "Bibliography", "Works Cited") or as a
      footnote whose body is a citation.
      EMIT ONE NODE PER CITED WORK. This is the one place you split a block:
      where a run of entries arrives as a single paragraph or list — with no
      blank line between them, or several packed onto one line — emit each
      work as its own node, cutting where one work's citation ends and the
      next author's name begins. Never merge two works into one node.
      Give each node that work's entry text as written, including its
      leading marker if it has one. Prose that merely mentions a work in
      passing ("as Pólya showed") is a paragraph, not a bibliographic node.
    - note: An authorial note bound to the body by a reference marker and
      printed outside the running text — a footnote at the foot of the page,
      an endnote, a margin note. It carries a marker (a superscript number, or
      a symbol such as *, †, ‡) that matches one in the body, and it says
      something about the SUBJECT: an aside, a caveat, a definition of a term
      used above, a remark on who a result is named after.
      Keep the marker in the content as written. Emit one note per marker.
      Bibliographic wins over note: a footnote whose body is a citation of an
      external work is bibliographic, not note.
      A "Note:", "Tip:", or "Warning:" callout sitting IN the running text is
      a paragraph, not a note — the test is the marker and the placement
      outside the body flow, not the word.
    - furniture: Page apparatus — text that belongs to the artifact rather
      than to what the document says. Running heads and running feet, folios
      (bare page numbers), the book or chapter title repeated at the top or
      bottom of the page, a colophon, a publisher or licence line, a "printed
      from" or "access this book at" line, marginal labels.
      The test is what the text is ABOUT: furniture describes the book as an
      object — its title, section, page, publisher, licence, URL — and would
      be equally true printed on any page. Everything that says something
      about the subject matter is not furniture. It usually sits at the very
      top or the very bottom of the page and repeats on every page.
      A furniture block is emitted as its own node, never folded into a
      neighbouring block. A run of apparatus is ONE furniture block even when
      it mixes text with a logo, badge, or image placeholder — a licence line
      like "Free PDF version ![Creative Commons License]() CC BY-NC-SA" is a
      single furniture node. Do not split it into pieces and do not re-type a
      piece by its shape: a placeholder that is part of the apparatus is
      furniture, not an image.
      This reaches INTO a line, never across blocks. A placeholder is part of
      the apparatus when it sits inside a run of apparatus text, as the licence
      line above has it. A placeholder standing alone as its own block — blank
      lines above and below — is an extracted figure and stays an image node,
      even on a title or copyright page where everything around it is
      apparatus.

      NOT furniture, whatever their position on the page:
      * A footnote. Its marker makes it look like apparatus, but it says
        something about the subject and the body refers to it. Type it as
        note, or as bibliographic when it cites a work.
      * A real heading. A heading that introduces content ON THIS PAGE is a
        header, even when it reads exactly like the running head — a page
        may show the same words twice, once as apparatus and once as the
        genuine section title. If only one such line is present and you
        cannot tell which it is, treat it as a header.
      * Captions, figure labels, table titles, and reference-list entries.
      * A fragment at the very start or end of the page. A bare number there
        looks exactly like a folio and is not one — it is the tail of
        something the previous page began. See the boundary rule above: it is
        content and must be emitted.

      WHEN IN DOUBT, DO NOT USE THIS TYPE. Give the block its ordinary type
      instead. Furniture is the one type that is discarded rather than kept,
      so a wrong call here deletes real content, while a missed one merely
      leaves a tidy line in the document."""

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
        recorder.record_example('extractor', {'segment_markdown': segment_markdown}, result)
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
        kept, discarded = _partition(extracted)
        for block in discarded:
            logger.debug(
                'page %d: discarded %s block %r',
                segment.index,
                block.type,
                logs.elide(block.content),
            )
        if discarded:
            logger.info(
                'extractor: page %d dropped %d furniture block(s)',
                segment.index,
                len(discarded),
            )
        nodes = [_node_for(block.type, block.content) for block in kept]
        return {'extract_results': [(segment.index, nodes)]}

    def collect(self, state: state.State) -> dict:
        """Merge each segment's extracted nodes back into the backbone.

        Args:
            state: The pipeline state, holding the extraction results.

        Returns:
            The updated segment backbone.
        """
        results = state.get('extract_results', [])
        segments = models.merge_results_into_segments(state['segments'], results, 'nodes')
        logger.info(
            'extractor: %d page(s) -> %d node(s)',
            len(results),
            sum(len(nodes) for _, nodes in results),
        )
        return {'segments': segments}
