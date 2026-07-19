import dspy

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
