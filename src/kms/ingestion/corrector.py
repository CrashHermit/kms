r"""
Correction pass for the Mistral OCR front-end.

Mistral transcribes a page faithfully most of the time, but it makes occasional
subtle, meaning-changing math errors that survive at any input resolution — e.g.
reading a plain radical `√` as an indexed root `∛`, or attaching a subscript to the
wrong symbol (`f(x)_1` for `f(x_1)`). Those are exactly the errors that silently
corrupt a math knowledge graph, and they are hard to catch without the source image.

This stage is a **generate-then-verify** second pass: a strong vision model re-reads
the page image alongside Mistral's markdown and fixes genuine transcription errors —
a verification task, which is easier and more reliable than transcribing from scratch,
so the checker stays away from its own OCR failure modes. It was validated to fix the
known error modes while leaving already-correct pages byte-identical.

Every transcribed page is proofread — not only math-bearing ones — since transcription
errors also appear in prose (dropped words, wrong characters, stray page chrome), and
proofreading the whole page keeps the always-rewrite model uniform and simple.

A **divergence guard** (`_within_tolerance`) keeps it safe: a "correction" whose length
swings far from the original is treated as a runaway rewrite (or a truncation) and
rejected — we keep the original transcription rather than trust a wholesale rewrite. Real
fixes are small.

The corrector also **normalizes math delimiters** to the pipeline's dollar convention so the
extractor and every downstream stage see uniform math. The unambiguous escape-sequence
delimiters (`\[ … \]`, `\( … \)`) are swapped deterministically by `_normalize_math_delimiters`
on every page (whether or not the vision correction was accepted); wrapping display equations the
OCR left *undelimited* needs to know what is display math, so that is asked of the vision model in
the prompt.

The corrector is always-rewrite: it returns the whole corrected page. A cheaper
conditional-output variant (emit a sentinel when the page is already clean, to skip the
rewrite output) is a drop-in future optimization behind the same interface.
"""

import base64
from pathlib import Path

import dspy
from langgraph.types import Send

from kms.core import tracing
from kms.core.llm import corrector_lm
from kms.core.models import Segment, merge_results_into_segments
from kms.core.state import State

# A correction should be a light edit; reject anything outside this band of the
# original length as a runaway rewrite or a truncation.
_TOLERANCE = 0.30


def _load_dspy_image(path: str | None) -> dspy.Image | None:
    """Load a PNG from disk into a dspy.Image (base64 data URL), or None if no path.

    The corrector is the only stage that needs a page image at the LLM boundary, so this
    dspy-specific helper lives here rather than in the (dspy-free) core models."""
    if not path:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return dspy.Image(url=f"data:image/png;base64,{encoded}")


def _within_tolerance(original: str, corrected: str) -> bool:
    """True when `corrected` is a plausible light edit of `original` (non-empty and within
    ±_TOLERANCE of its length). Guards against the corrector truncating or wholesale
    rewriting the page."""
    if not corrected or not corrected.strip():
        return False
    lo, hi = len(original) * (1 - _TOLERANCE), len(original) * (1 + _TOLERANCE)
    return lo <= len(corrected) <= hi


# LaTeX math-delimiter escape sequences → the pipeline's dollar convention. `\[`/`\]` and
# `\(`/`\)` are unambiguous math delimiters (they do not occur in prose), so a straight,
# whitespace-preserving token swap is safe and deterministic.
_DELIMITER_SWAPS = ((r"\[", "$$"), (r"\]", "$$"), (r"\(", "$"), (r"\)", "$"))


def _normalize_math_delimiters(text: str) -> str:
    """Rewrite LaTeX math delimiters to `$`/`$$`: `\\[ … \\]` → `$$ … $$` (display) and
    `\\( … \\)` → `$ … $` (inline). Runs on every proofread page — whether or not the vision
    correction was accepted — so display math is uniform for the extractor and downstream
    stages. Bare, *undelimited* display blocks are handled in the prompt, not here."""
    for old, new in _DELIMITER_SWAPS:
        text = text.replace(old, new)
    return text


class Signature(dspy.Signature):
    r"""
    You are a meticulous mathematics proofreader. You are given the image of a single
    textbook page and an OCR transcription of that page in markdown. Compare them and
    return a corrected transcription.

    Correct ONLY genuine transcription errors — do not rewrite, restructure, reformat,
    or re-transcribe text that already matches the image (the one deliberate exception is
    math-delimiter normalization, described under LATEX FORMAT below). Preserve the
    transcription's wording, structure, and markdown exactly except where it disagrees
    with the image.

    Scrutinize mathematical notation token by token against the image, since that is
    where transcription errors hide:
    - root indices: a plain square root `\sqrt{x}` must NOT gain an index (`\sqrt[3]{x}`),
      and an indexed root must keep its true index;
    - subscripts/superscripts: attach each to the correct symbol (`f(x_1)`, not `f(x)_1`);
    - operators, relations, delimiters, and Greek letters.

    LATEX FORMAT — keep all math in LaTeX and normalize its delimiters to dollar signs:
    inline math in single dollars `$ … $`, display math in double `$$ … $$`. This delimiter
    normalization is required (the one allowed exception to "do not reformat"); change only
    the delimiters, never the math content:
    - convert `\( … \)` to `$ … $` and `\[ … \]` to `$$ … $$`;
    - wrap any display equation the transcription left undelimited — a standalone equation
      line, or a bare `\begin{array}` / `aligned` / `cases` / `equation` block — in `$$ … $$`.

    Return the full corrected markdown for the page and nothing else. If the transcription
    is already faithful (apart from any delimiter normalization above), return it unchanged.
    """

    page_image: dspy.Image = dspy.InputField(
        description="The image of the textbook page — the ground truth to check the transcription against."
    )
    transcription: str = dspy.InputField(
        description="The OCR markdown transcription of the page to proofread."
    )
    corrected: str = dspy.OutputField(
        description="The full corrected markdown transcription of the page, with only genuine errors fixed."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.proofreader = dspy.Predict(Signature)
        self.set_lm(lm or corrector_lm())

    async def aforward(self, page_image: dspy.Image, transcription: str) -> str:
        result = await self.proofreader.acall(page_image=page_image, transcription=transcription)
        corrected = result.corrected or ""
        # The page image is intentionally not captured (large base64, and it is
        # reconstructable from the input PDF); the trainable text signal is the
        # transcription -> corrected pair.
        tracing.record(
            "corrector",
            inputs={"transcription": transcription, "page_image": "<omitted: page render>"},
            outputs={"corrected": corrected},
        )
        return corrected


# --- LangGraph node: proofread each Mistral-transcribed page against its image ---


class CorrectorNode:
    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per transcribed segment. Every page is proofread; a segment
        with no content or no page image to check against is skipped, and if none qualify
        the stage is a no-op."""
        segments = state.get("segments", [])
        sends = [
            Send("corrector_worker", {"segment": seg})
            for seg in segments
            if seg.content and seg.image_path
        ]
        return sends or "corrector_collect"

    async def worker(self, state: dict) -> dict:
        """Proofread one page's transcription against its image, keeping the original if
        the correction diverges too far (runaway rewrite / truncation)."""
        segment: Segment = state["segment"]
        corrected = await self.module.aforward(
            page_image=_load_dspy_image(segment.image_path),
            transcription=segment.content,
        )
        final = corrected if _within_tolerance(segment.content, corrected) else segment.content
        # Normalize math delimiters on the chosen text — even when the correction was
        # rejected, so a kept-original page still gets uniform `$$`/`$` delimiters.
        final = _normalize_math_delimiters(final)
        return {"correction_results": [(segment.index, final)]}

    def collect(self, state: State) -> dict:
        """Write each corrected transcription back into its segment. Segments that were
        not dispatched keep their original content untouched."""
        segments = merge_results_into_segments(
            state["segments"], state.get("correction_results", []), "content"
        )
        return {"segments": segments}
