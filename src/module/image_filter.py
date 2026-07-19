import dspy
from langgraph.types import Send

from .state import State, Segment, Picture, load_dspy_image


class Signature(dspy.Signature):
    """
    You are an expert at identifying whether a picture extracted from a technical
    textbook is substantive and worth keeping versus decorative or navigational
    noise that should be discarded.

    You are given the full parent node image for context and the individual picture
    image to evaluate. Use the parent node image to understand where the picture
    sits in the flow of the content and what role it plays — a picture that looks
    ambiguous in isolation is often clearly decorative or clearly substantive when
    seen in context.

    ## KEEP — Substantive Pictures

    Keep a picture if it carries information that a reader would lose if the
    picture were removed. This includes:

    - Diagrams, charts, graphs, and plots
    - Technical illustrations and labeled figures
    - Photographs relevant to the subject matter
    - Mathematical expressions or equations captured as images rather than text
    - Tables or structured data rendered as images
    - Any visual that is referenced by the surrounding text (e.g. "see Figure 3.2")

    ## DISCARD — Non-Substantive Pictures

    Discard a picture if it is purely decorative, structural, or navigational
    and carries no informational content. This includes:

    - Decorative icons, ornamental graphics, or visual flourishes
    - Publisher logos, watermarks, or branding elements
    - Horizontal rules, dividers, or border graphics
    - Navigation arrows, buttons, or UI chrome
    - Repeated header or footer graphics
    - Small bullets or list markers captured as images

    ## WHEN IN DOUBT

    If you are uncertain whether a picture is substantive, keep it. It is better
    to retain a borderline picture than to discard something a reader might need.

    LATEX FORMAT:
    When discussing or preserving mathematical notation, use LaTeX delimiters:
    `$ $` for inline math and `$$ $$` for block/display math.
    """

    parent_image: dspy.Image = dspy.InputField(
        description=(
            "The full parent node image. Use this to understand the visual context "
            "and role of the picture — where it sits on the segment, what surrounds "
            "it, and whether the text references it."
        )
    )

    image_image: dspy.Image = dspy.InputField(
        description=(
            "The individual image extracted from the parent node. "
            "Evaluate this against the keep/discard criteria above."
        )
    )

    is_substantive: bool = dspy.OutputField(
        description=(
            "True if the image should be kept (it carries meaningful information), "
            "False if it should be discarded (decorative or navigational noise)."
        )
    )


class Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(Signature)

    async def aforward(
        self,
        parent_image: dspy.Image,
        image_image: dspy.Image,
    ):
        result = await self.classifier.acall(
            parent_image=parent_image,
            image_image=image_image,
        )
        return dspy.Prediction(is_substantive=result.is_substantive)


# --- LangGraph node: keep substantive pictures, drop noise ---

_module = Module()


def dispatch(state: State) -> list[Send] | str:
    """Fan out one worker per segment that has pictures to evaluate."""
    segments = state.get("segments", [])
    sends = [Send("image_filter_worker", {"segment": seg}) for seg in segments if seg.pictures]
    return sends or "image_filter_collect"


async def image_filter_worker(state: dict) -> dict:
    """Classify each picture on one segment, keeping only the substantive ones."""
    segment: Segment = state["segment"]
    parent_image = load_dspy_image(segment.image_path)

    kept: list[Picture] = []
    for picture in segment.pictures:
        prediction = await _module.aforward(
            parent_image=parent_image,
            image_image=load_dspy_image(picture.image_path),
        )
        if prediction.is_substantive:
            kept.append(picture)

    return {"filter_results": [(segment.index, kept)]}


def image_filter_collect(state: State) -> dict:
    """Merge each segment's surviving pictures back into the ordered backbone."""
    kept_by_index = dict(state.get("filter_results", []))
    for segment in state["segments"]:
        if segment.index in kept_by_index:
            segment.pictures = kept_by_index[segment.index]
    return {"segments": state["segments"]}
