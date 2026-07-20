import re

import dspy
from langgraph.types import Send

from .state import State, Segment, load_dspy_image
from .llm import vision_lm

# Tolerant match for an image placeholder: ![N]() with optional surrounding whitespace.
_PLACEHOLDER = re.compile(r"!\[\s*\d+\s*\]\(\s*\)")


def _reconcile_placeholders(content: str, num_pictures: int) -> str:
    """Relabel image placeholders to a contiguous 1..N by order of appearance.

    OCR is asked to number pictures 1..N in reading order, but the vision model is
    unreliable at it — it reuses a number for two figures or emits one past the
    surviving count. Rather than trust those numbers, renumber the ![N]()
    placeholders positionally to match the pictures that survived filtering
    (segment.pictures, already 1..N), and drop any extras beyond the count.
    """
    counter = 0

    def repl(_match: re.Match) -> str:
        nonlocal counter
        counter += 1
        return f"![{counter}]()" if counter <= num_pictures else ""

    return _PLACEHOLDER.sub(repl, content)


class Signature(dspy.Signature):
    r"""
    You are an Expert Technical Textbook OCR Engine.

    You receive an image of a single textbook node. Your only job is to
    faithfully transcribe the node into clean, well-formed markdown. Do not
    attempt to classify, restructure, or interpret the content — just convert
    what you see as accurately as possible.

    ## OUTPUT FORMAT

    Output a single `content` string of well-formed markdown that flows in
    reading order.

    ## GENERAL TRANSCRIPTION RULES

    - Preserve all headings using standard markdown heading levels (`#`, `##`, etc.).
    - Preserve all prose paragraphs as plain markdown paragraphs.
    - Preserve all lists as markdown bullet or ordered lists.
    - Preserve all tables as markdown tables.
    - All mathematical notation must use LaTeX format. Use single dollar signs `$ $` for inline math and double dollar signs `$$ $$` for block/display math. Multi-line display math: `$$\begin{aligned} ... \end{aligned}$$`.
    - Code: fenced blocks with the appropriate language tag.
    - Skip running headers, footers, node numbers, and repeated chapter/book
      titles — these are navigational chrome, not content.

    ## READING ORDER — TRANSCRIBE EACH ELEMENT EXACTLY ONCE

    - Follow a single reading order: top to bottom, and for a multi-column or boxed
      layout, read each column or box fully in order. Do NOT jump between columns
      mid-paragraph.
    - Transcribe every element (paragraph, list item, numbered exercise/checkpoint,
      figure, equation) exactly ONCE. Never repeat or re-emit content you have already
      transcribed, even when it sits in a sidebar, callout box, or spans a column break.
      Duplicated blocks are a serious error.

    ## THEOREMS, PROOFS, AND WORKED SOLUTIONS — KEEP THE BLOCKS SEPARATE

    - Keep a labelled statement (e.g. `Theorem 3.2`, `Proposition 15.7`, `Definition`,
      `Example 4`) together with the statement text that follows it, as one block.
    - When a proof or worked solution follows, PRESERVE its opening marker (`Proof.`,
      `Solution.`, and the closing `∎`/`□` if present) and start the proof/solution on a
      NEW paragraph, visually separated from the statement — even when the book runs
      `Proof.` inline immediately after the statement. The statement and its
      proof/solution must be DISTINCT blocks, never merged into one paragraph.

    ## PARTIAL CONTENT AT node BOUNDARIES

    - If the node begins mid-sentence, mid-equation, mid-table, or mid-block,
      transcribe it exactly as it appears — starting from wherever the node starts.
      Do not attempt to reconstruct what came before.
    - If the node ends mid-sentence, mid-equation, mid-table, or mid-block,
      transcribe up to exactly where the node ends. Do not attempt to complete it.
    - Faithfully represent what is physically on this node, nothing more.

    ## IMAGES & FIGURES

    - For each entry in `current_node_pictures`, insert a placeholder at the
      visual location where the image appears in the node flow:
      `![{N}]()`
      where `{N}` is the corresponding value from `current_node_picture_indices`.
      These indices run 1, 2, 3, … in the same order as `current_node_pictures`
      and the reading order of the images on the page, so the first image is
      `![1]()`, the second `![2]()`, and so on.
    - Place the caption or figure label (if any) as plain text immediately after
      the placeholder, e.g.:
      ![1]()
      **Figure 3.2** The unit circle with key angles labeled.
    - If a figure has no caption, just emit the placeholder alone.


    ## SIDEBARS & MARGINAL CONTENT

    - Textbooks often have sidebars, margin notes, biographical blurbs, "Did You
      Know?" boxes, and similar off-flow content. Integrate this content naturally
      into the flow of the markdown where it fits best — typically just before or
      after the paragraph it annotates. Treat it as regular prose, no special
      formatting needed.
    """

    current_node_image: dspy.Image = dspy.InputField(
        description="The image of the node to transcribe. Extract content from this node only, exactly as it appears."
    )
    current_node_pictures: list[dspy.Image] | None = dspy.InputField(
        description="The images of the pictures in the current node, in order."
    )
    current_node_picture_indices: list[int] = dspy.InputField(
        description="The number to use for each picture in current_node_pictures, in the same order — running 1, 2, 3, … in reading order. Use these for the ![N]() placeholders."
    )

    current_node_content: str = dspy.OutputField(
        description="The faithfully transcribed markdown content of the current node, exactly as it appears — including any partial blocks at the top or bottom."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.transcriber = dspy.ChainOfThought(Signature)
        self.set_lm(lm or vision_lm())

    async def aforward(
        self,
        current_node_image: dspy.Image,
        current_node_picture_indices: list[int],
        current_node_pictures: list[dspy.Image] | None = None,
    ):
        result = await self.transcriber.acall(
            current_node_image=current_node_image,
            current_node_pictures=current_node_pictures,
            current_node_picture_indices=current_node_picture_indices,
        )
        return dspy.Prediction(current_node_content=result.current_node_content)


# --- LangGraph node: transcribe each segment page to markdown ---

class OCRNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per segment. Each page is transcribed in isolation: a node
        split across a page break is healed downstream by the seam merger, which works on
        the extracted node text rather than the page image, so neighbour page images buy
        the transcriber nothing here — dropping them cuts the per-call image payload from
        three pages to one."""
        segments = state.get("segments", [])
        if not segments:
            return "ocr_collect"
        return [Send("ocr_worker", {"segment": seg}) for seg in segments]

    async def worker(self, state: dict) -> dict:
        """Transcribe one segment's page image into markdown."""
        segment: Segment = state["segment"]
        pictures = segment.pictures
        prediction = await self.module.aforward(
            current_node_image=load_dspy_image(segment.image_path),
            current_node_picture_indices=[p.index for p in pictures],
            current_node_pictures=[load_dspy_image(p.image_path) for p in pictures] or None,
        )
        return {"ocr_results": [(segment.index, prediction.current_node_content)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's transcribed markdown back into the ordered backbone,
        reconciling its image placeholders against the surviving pictures."""
        content_by_index = dict(state.get("ocr_results", []))
        for segment in state["segments"]:
            if segment.index in content_by_index:
                segment.content = _reconcile_placeholders(
                    content_by_index[segment.index], len(segment.pictures)
                )
        return {"segments": state["segments"]}
