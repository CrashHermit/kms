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

import asyncio
import base64
import logging
import re
from pathlib import Path

import dspy
from langgraph.types import Send

from kms.core import llm, models, state

logger = logging.getLogger(__name__)

# A correction should be a light edit; reject anything outside this band of the
# original length as a runaway rewrite or a truncation.
_TOLERANCE = 0.30


def _load_dspy_image(path: str | None) -> dspy.Image | None:
    """Load a PNG from disk into a dspy.Image (base64 data URL), or None if no path.

    The corrector is the only stage that needs a page image at the LLM boundary, so this
    dspy-specific helper lives here rather than in the (dspy-free) core models."""
    if not path:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


def _within_tolerance(original: str, corrected: str) -> bool:
    """True when `corrected` is a plausible light edit of `original` (non-empty and within
    ±_TOLERANCE of its length). Guards against the corrector truncating or wholesale
    rewriting the page."""
    if not corrected or not corrected.strip():
        return False
    lo, hi = len(original) * (1 - _TOLERANCE), len(original) * (1 + _TOLERANCE)
    return lo <= len(corrected) <= hi


# LaTeX math-delimiter escape sequences → the pipeline's dollar convention. `\[`/`\]` and
# `\(`/`\)` are unambiguous math delimiters (they do not occur in prose), so a straight
# token swap is safe and deterministic.
_DELIMITER_SWAPS = ((r'\[', '$$'), (r'\]', '$$'), (r'\(', '$'), (r'\)', '$'))

# A `$$ … $$` or `$ … $` span, used to strip padding from inside the delimiters. Display
# is matched first so a display pair is never read as two empty inline ones.
_MATH_SPAN = re.compile(r'\$\$(.*?)\$\$|\$(.*?)\$', re.DOTALL)


def _strip_delimiter_padding(match: re.Match) -> str:
    """Rewrite one math span with no spaces or tabs immediately inside its delimiters.

    Newlines are kept: a wrapped display block is conventionally written `$$\\n … \\n$$`,
    and collapsing it onto one line would reflow the page for no gain. Only the
    horizontal padding the escape-form swap introduces is removed."""
    display, inline = match.group(1), match.group(2)
    body = display if display is not None else inline
    fence = '$$' if display is not None else '$'
    stripped = body.strip(' \t')
    return f'{fence}{stripped}{fence}' if stripped else match.group(0)


def _normalize_math_delimiters(text: str) -> str:
    """Rewrite LaTeX math delimiters to `$`/`$$`: `\\[ … \\]` → `$$…$$` (display) and
    `\\( … \\)` → `$…$` (inline). Runs on every proofread page — whether or not the vision
    correction was accepted — so display math is uniform for the extractor and downstream
    stages. Bare, *undelimited* display blocks are handled in the prompt, not here.

    Padding inside the delimiters is stripped as part of the swap. Mistral writes the
    escape forms with spaces (`\\( 2^p-1 \\)`), and a literal swap would carry them into
    `$ 2^p-1 $` — a second spelling of every inline expression, differing from the
    unpadded `$2^p-1$` the same OCR emits elsewhere on the same page. Downstream stages
    then have to match both, and the corrector's training labels have to pick one
    arbitrarily. Whitespace immediately inside a math delimiter is never significant to
    LaTeX, so collapsing it loses nothing."""
    if not any(token in text for token, _ in _DELIMITER_SWAPS):
        # No escape-form delimiters, so no padding was introduced and there is nothing to
        # tidy. Returning early confines the span rewrite to the case that motivates it,
        # rather than letting it walk every `$`-to-`$` run on a page — which on prose
        # containing two currency amounts would pair `$5 … $6` as if it were math.
        return text
    for old, new in _DELIMITER_SWAPS:
        text = text.replace(old, new)
    return _MATH_SPAN.sub(_strip_delimiter_padding, text)


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
      line, or a bare `\begin{array}` / `aligned` / `cases` / `equation` block — in `$$ … $$`;
    - rewrite math the transcription left as plain Unicode text into LaTeX inside dollars:
      `x² − 1` becomes `$x^2 - 1$`, `n ≥ 3` becomes `$n \geq 3$`, `qʳ⁺¹` becomes
      `$q^{r+1}$`. Operators (`×÷±≤≥≠∈⊂∪∩→∞√∑∫`) and super/subscript digits (`²`, `⁵`,
      `ₙ`) must not survive outside a math span. Wrap the expression, never reword it.

    ENCODING — the page's characters, written the way the pipeline expects them:
    - part-labels go in parentheses: a label printed as a circled letter (`ⓐ`, `ⓑ`) is
      written `(a)`, `(b)`;
    - quotation marks and apostrophes are ASCII `"` and `'`, never curly.
    These are re-encodings of what the image shows, not rewordings — never change, add, or
    drop a word to satisfy them.

    Do NOT add markup the transcription does not already have. If the OCR dropped a bold
    label or an italic defined term, leave it dropped: restoring emphasis is reformatting,
    and it is the first step from proofreading toward rewriting.

    Return the full corrected markdown for the page and nothing else. If the transcription
    is already faithful (apart from the normalizations above), return it unchanged.
    """

    page_image: dspy.Image = dspy.InputField(
        description='The image of the textbook page — the ground truth to check the transcription against.'
    )
    transcription: str = dspy.InputField(
        description='The OCR markdown transcription of the page to proofread.'
    )
    corrected: str = dspy.OutputField(
        description='The full corrected markdown transcription of the page, with only genuine errors fixed.'
    )


class Corrector(dspy.Module):
    """Proofreads a transcribed page against its source image and fixes OCR errors."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.proofreader = dspy.Predict(Signature)
        self.set_lm(language_model or llm.corrector_lm())

    async def aforward(self, page_image: dspy.Image, transcription: str) -> str:
        """Returns the proofread transcription with genuine errors corrected."""
        result = await self.proofreader.acall(
            page_image=page_image, transcription=transcription
        )
        corrected = result.corrected or ''
        logger.debug(
            'proofread: %d chars in, %d chars out',
            len(transcription),
            len(corrected),
        )
        return corrected

    def forward(self, page_image: dspy.Image, transcription: str) -> str:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(page_image, transcription))


# --- LangGraph node: proofread each Mistral-transcribed page against its image ---


class CorrectorNode:
    """LangGraph node: fans out per-page proofreaders and collects the corrected text."""

    def __init__(self, module: Corrector | None = None) -> None:
        self.module = module or Corrector()

    def dispatch(self, state: state.State) -> list[Send] | str:
        """Fan out one worker per transcribed segment. Every page is proofread; a segment
        with no content or no page image to check against is skipped, and if none qualify
        the stage is a no-op."""
        segments = state.get('segments', [])
        sends = [
            Send('corrector_worker', {'segment': segment})
            for segment in segments
            if segment.content and segment.image_path
        ]
        return sends or 'corrector_collect'

    async def worker(self, state: dict) -> dict:
        """Proofread one page's transcription against its image, keeping the original if
        the correction diverges too far (runaway rewrite / truncation)."""
        segment: models.Segment = state['segment']
        corrected = await self.module.aforward(
            page_image=_load_dspy_image(segment.image_path),
            transcription=segment.content,
        )
        kept = _within_tolerance(segment.content, corrected)
        if not kept:
            # A runaway rewrite or a truncated completion; the page silently keeps its
            # original transcription, so this is the only signal it happened.
            logger.warning(
                'page %d: correction rejected (%d chars in, %d out); keeping '
                'the original transcription',
                segment.index,
                len(segment.content or ''),
                len(corrected),
            )
        final = corrected if kept else segment.content
        # Normalize math delimiters on the chosen text — even when the correction was
        # rejected, so a kept-original page still gets uniform `$$`/`$` delimiters.
        final = _normalize_math_delimiters(final)
        return {'correction_results': [(segment.index, final)]}

    def collect(self, state: state.State) -> dict:
        """Write each corrected transcription back into its segment. Segments that were
        not dispatched keep their original content untouched."""
        results = state.get('correction_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'content'
        )
        logger.info('corrector: %d page(s) proofread', len(results))
        return {'segments': segments}
