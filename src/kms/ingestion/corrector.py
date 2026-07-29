r"""
Correction pass for the Mistral OCR front-end.

Mistral transcribes a page faithfully most of the time, but it makes occasional
subtle, meaning-changing math errors that survive at any input resolution — e.g.
reading a plain radical `√` as an indexed root `∛`, or attaching a subscript to
the wrong symbol (`f(x)_1` for `f(x_1)`). Those are exactly the errors that
silently corrupt a math knowledge graph, and they are hard to catch without the
source image.

This stage is a **generate-then-verify** second pass: a strong vision model
re-reads the page image alongside Mistral's markdown and fixes genuine
transcription errors — a verification task, which is easier and more reliable
than transcribing from scratch, so the checker stays away from its own OCR
failure modes. It was validated to fix the known error modes while leaving
already-correct pages byte-identical.

Every transcribed page is proofread — not only math-bearing ones — since
transcription errors also appear in prose (dropped words, wrong characters,
stray page chrome), and proofreading the whole page keeps the always-rewrite
model uniform and simple.

The stage is **unguarded**: whatever the model returns is what the page becomes.
There is deliberately no divergence check on the correction — the pipeline's
passes are bare-bones LLM calls, and a page's correctness rests on the prompt
rather than on a wrapper second-guessing the output.

The corrector also **normalizes math delimiters** to the pipeline's dollar
convention so the extractor and every downstream stage see uniform math. The
unambiguous escape-sequence delimiters (`\[ … \]`, `\( … \)`) are swapped
deterministically by `_normalize_math_delimiters`; wrapping display equations
the OCR left *undelimited* needs to know what is display math, so that is asked
of the vision model in the prompt.

The corrector is always-rewrite: it returns the whole corrected page. A cheaper
conditional-output variant (emit a sentinel when the page is already clean, to
skip the rewrite output) is a drop-in future optimization behind the same
interface.
"""

import asyncio
import base64
import logging
from pathlib import Path

import dspy
from langgraph.types import Send

from kms.core import llm, models, state

logger = logging.getLogger(__name__)


def _load_dspy_image(path: str | None) -> dspy.Image | None:
    """Load a PNG from disk into a ``dspy.Image`` (base64 data URL).

    The corrector is the only stage that needs a page image at the LLM
    boundary, so this dspy-specific helper lives here rather than in the
    (dspy-free) core models.

    Args:
        path: The page render's path, or None.

    Returns:
        The loaded image, or None when no path was given.
    """
    if not path:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


# LaTeX math-delimiter escape sequences → the pipeline's dollar convention.
# `\[`/`\]` and `\(`/`\)` are unambiguous math delimiters (they do not occur in
# prose), so a straight, whitespace-preserving token swap is safe and
# deterministic.
_DELIMITER_SWAPS = ((r'\[', '$$'), (r'\]', '$$'), (r'\(', '$'), (r'\)', '$'))


def _normalize_math_delimiters(text: str) -> str:
    """Rewrite LaTeX math delimiters to the pipeline's dollar convention.

    ``\\[ … \\]`` becomes ``$$ … $$`` (display) and ``\\( … \\)`` becomes
    ``$ … $`` (inline). Runs on every proofread page, so display math is
    uniform for the extractor and downstream stages. Bare, *undelimited*
    display blocks are handled in the prompt, not here.

    Args:
        text: The page's markdown.

    Returns:
        The markdown with dollar-delimited math.
    """
    for old, new in _DELIMITER_SWAPS:
        text = text.replace(old, new)
    return text


class Signature(dspy.Signature):
    r"""
    You are a meticulous mathematics proofreader. You are given the image of a
    single textbook page and an OCR transcription of that page in markdown.
    Compare them and return a corrected transcription.

    Correct ONLY genuine transcription errors — do not rewrite, restructure,
    reformat, or re-transcribe text that already matches the image (the one
    deliberate exception is math-delimiter normalization, described under LATEX
    FORMAT below). Preserve the transcription's wording, structure, and markdown
    exactly except where it disagrees with the image.

    Scrutinize mathematical notation token by token against the image, since
    that is where transcription errors hide:
    - root indices: a plain square root `\sqrt{x}` must NOT gain an index
      (`\sqrt[3]{x}`), and an indexed root must keep its true index;
    - subscripts/superscripts: attach each to the correct symbol (`f(x_1)`, not
      `f(x)_1`);
    - operators, relations, delimiters, and Greek letters.

    LATEX FORMAT — keep all math in LaTeX and normalize its delimiters to dollar
    signs: inline math in single dollars `$ … $`, display math in double
    `$$ … $$`. This delimiter normalization is required (the one allowed
    exception to "do not reformat"); change only the delimiters, never the math
    content:
    - convert `\( … \)` to `$ … $` and `\[ … \]` to `$$ … $$`;
    - wrap any display equation the transcription left undelimited — a
      standalone equation line, or a bare `\begin{array}` / `aligned` / `cases`
      / `equation` block — in `$$ … $$`.

    Return the full corrected markdown for the page and nothing else. If the
    transcription is already faithful (apart from any delimiter normalization
    above), return it unchanged.
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
    """Proofreads a transcribed page against its source image.

    Args:
        language_model: The vision LM to run on. Defaults to
            ``llm.corrector_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.proofreader = dspy.Predict(Signature)
        self.set_lm(language_model or llm.corrector_lm())

    async def aforward(self, page_image: dspy.Image, transcription: str) -> str:
        """Proofread one page.

        Args:
            page_image: The page render to check against.
            transcription: The OCR markdown to proofread.

        Returns:
            The proofread transcription, with genuine errors corrected.
        """
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


# --- LangGraph node: proofread each transcribed page against its image ---


class CorrectorNode:
    """Fans out per-page proofreaders and collects the corrected text.

    Args:
        module: The proofreading module. Created fresh if None.
    """

    def __init__(self, module: Corrector | None = None) -> None:
        self.module = module or Corrector()

    def dispatch(self, state: state.State) -> list[Send] | str:
        """Fan out one worker per transcribed segment.

        Every page is proofread; a segment with no content or no page image to
        check against is skipped.

        Args:
            state: The pipeline state, holding the segment backbone.

        Returns:
            One Send per qualifying segment, or the collect step's name when
            none qualify (the stage is then a no-op).
        """
        segments = state.get('segments', [])
        sends = [
            Send('corrector_worker', {'segment': segment})
            for segment in segments
            if segment.content and segment.image_path
        ]
        return sends or 'corrector_collect'

    async def worker(self, state: dict) -> dict:
        """Proofread one page's transcription against its image.

        The correction is taken as returned — the page becomes whatever the
        model produced, with only its math delimiters normalized.

        Args:
            state: The worker payload, holding its ``segment``.

        Returns:
            The page's ``correction_results`` entry.
        """
        segment: models.Segment = state['segment']
        corrected = await self.module.aforward(
            page_image=_load_dspy_image(segment.image_path),
            transcription=segment.content,
        )
        return {
            'correction_results': [
                (segment.index, _normalize_math_delimiters(corrected))
            ]
        }

    def collect(self, state: state.State) -> dict:
        """Write each corrected transcription back into its segment.

        Segments that were not dispatched keep their original content
        untouched.

        Args:
            state: The pipeline state, holding the correction results.

        Returns:
            The updated segment backbone.
        """
        results = state.get('correction_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'content'
        )
        logger.info('corrector: %d page(s) proofread', len(results))
        return {'segments': segments}
