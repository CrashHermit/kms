import dspy
from langgraph.types import Send

from .state import State, Segment, load_dspy_image
from .llm import vision_lm


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
      For example, if `current_node_picture_indices` is `[1, 3]`, use `![1]()` for
      the first image and `![3]()` for the second image.
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

    previous_node_image_context: dspy.Image | None = dspy.InputField(
        description="The image of the node immediately before the current node, if available. Use for context only — do not transcribe it."
    )
    current_node_image: dspy.Image = dspy.InputField(
        description="The image of the node to transcribe. Extract content from this node only, exactly as it appears."
    )
    next_node_image_context: dspy.Image | None = dspy.InputField(
        description="The image of the node immediately after the current node, if available. Use for context only — do not transcribe it."
    )
    current_node_pictures: list[dspy.Image] | None = dspy.InputField(
        description="The images of the pictures in the current node, in order."
    )
    current_node_picture_indices: list[int] = dspy.InputField(
        description="The actual image indices corresponding to each picture in current_node_pictures. Use these indices for the ![N]() placeholders, not sequential numbering."
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
        previous_node_image_context: dspy.Image | None = None,
        next_node_image_context: dspy.Image | None = None,
        current_node_pictures: list[dspy.Image] | None = None,
    ):
        result = await self.transcriber.acall(
            previous_node_image_context=previous_node_image_context,
            current_node_image=current_node_image,
            next_node_image_context=next_node_image_context,
            current_node_pictures=current_node_pictures,
            current_node_picture_indices=current_node_picture_indices,
        )
        return dspy.Prediction(current_node_content=result.current_node_content)


# --- LangGraph node: transcribe each segment page to markdown ---

class OCRNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per segment, passing its neighbours as read-only context."""
        segments = state.get("segments", [])
        if not segments:
            return "ocr_collect"
        return [
            Send("ocr_worker", {
                "segment": seg,
                "previous": segments[i - 1] if i > 0 else None,
                "next": segments[i + 1] if i < len(segments) - 1 else None,
            })
            for i, seg in enumerate(segments)
        ]

    async def worker(self, state: dict) -> dict:
        """Transcribe one segment's page image into markdown."""
        segment: Segment = state["segment"]
        previous: Segment | None = state.get("previous")
        following: Segment | None = state.get("next")

        pictures = segment.pictures
        prediction = await self.module.aforward(
            current_node_image=load_dspy_image(segment.image_path),
            current_node_picture_indices=[p.index for p in pictures],
            previous_node_image_context=load_dspy_image(previous.image_path) if previous else None,
            next_node_image_context=load_dspy_image(following.image_path) if following else None,
            current_node_pictures=[load_dspy_image(p.image_path) for p in pictures] or None,
        )

        return {"ocr_results": [(segment.index, prediction.current_node_content)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's transcribed markdown back into the ordered backbone."""
        content_by_index = dict(state.get("ocr_results", []))
        for segment in state["segments"]:
            if segment.index in content_by_index:
                segment.content = content_by_index[segment.index]
        return {"segments": state["segments"]}
