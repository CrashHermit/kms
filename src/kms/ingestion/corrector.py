r"""
Correction pass for the Mistral OCR front-end.

Mistral transcribes a page faithfully most of the time, but it makes occasional
subtle, meaning-changing errors that survive at any input resolution — reading a
plain radical `√` as an indexed root `∛`, attaching a subscript to the wrong
symbol (`f(x)_1` for `f(x_1)`), dropping a negation, shifting a table cell into
the next column. Those are exactly the errors that silently corrupt a knowledge
graph, and they are hard to catch without the source image.

This stage is a **generate-then-verify** second pass: a strong vision model
re-reads the page image alongside Mistral's markdown and fixes genuine
transcription errors — a verification task, which is easier and more reliable
than transcribing from scratch, so the checker stays away from its own OCR
failure modes.

The corrector is a **fidelity** pass, not a quality pass: its single question is
whether the transcription says what the page says. The image is therefore its
only authority, and any judgement the image cannot settle — what the subject
matter ought to say, which convention the document follows, what would read
better — is out of scope by construction. That is what keeps it from editing the
source's own errors or standardising its notation.

The prompt is **domain-general and stated as principles** rather than as a
checklist of known error shapes. Scrutiny is allocated by redundancy instead of
by subject: prose says the same thing several ways at once and can be read for
sense, while notation, identifiers, quantities, and tabular data carry no such
slack and are read character by character. The meaning-changing differences are
named as classes (attachment, extent, substitution, polarity, quantity,
relation, position, order, presence) so a non-mathematical source is handled by
the same pass.

The stage is **unguarded**: whatever the model returns is what the page becomes.
There is deliberately no divergence check on the correction — the pipeline's
passes are bare-bones LLM calls, and a page's correctness rests on the prompt
rather than on a wrapper second-guessing the output.

**Formatting is not the corrector's job** — including math delimiters. The
prompt tells it to leave markdown structure, delimiters, emphasis, and cosmetic
whitespace exactly as transcribed, so presentation is untouched here and belongs
to the formatter stage instead. Note the consequence while that stage does not
exist: nothing in this pass converts `\( … \)` / `\[ … \]` to the dollar
convention the extractor's prompt asks for.

Two boundaries of that rule are drawn explicitly, because both were found
undefined when the prompt was first evaluated against real OCR output:

- Whitespace that *carries meaning* is not formatting. Indentation inside a
  code block is structure — Mistral flattens it, and a line at the wrong depth
  says something the page does not — so it is corrected under the position
  class rather than frozen under the formatting prohibition.
- Page furniture is out of scope in **both** directions. Running heads, folios,
  and marginal labels are neither removed when present nor restored when
  Mistral drops them (it usually does). Whether chrome belongs in the document
  is a presentation policy, which is the formatter's to decide, not a fidelity
  question this pass can settle from the image. A footnote is explicitly *not*
  furniture — the front end now appends the page's footer back onto its
  markdown, so a citation of an external work arrives here as a trailing block,
  and the prompt says to proofread and keep it rather than read a page-bottom
  block as chrome to leave alone.

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

from kms.core import llm, models, recorder, state

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


class Signature(dspy.Signature):
    r"""
    You are a meticulous proofreader of OCR transcriptions. You are given the
    image of a single document page and an OCR transcription of that page in
    markdown. Compare them and return a corrected transcription.

    The image is your only authority. Correct a difference only when the image
    settles it. If deciding would take knowledge the image cannot give you —
    what the subject matter ought to say, which convention the document
    follows, what would read better — leave the transcription as it is.

    Correct differences that change meaning, and leave every other difference
    alone. This is a check on fidelity, not on quality.

    HOW CLOSELY TO READ

    Redundancy, not subject matter, decides how much scrutiny a passage needs.
    Prose says the same thing several ways at once, so context repairs a
    misread word and you can read it for sense. Notation, identifiers,
    quantities, code, and tabular data carry no such slack — `x_2` and `x^2`
    are equally plausible in isolation, and only the image tells you which was
    written. Read low-redundancy content character by character.

    WHAT COUNTS AS A MEANING-CHANGING DIFFERENCE

    - Attachment — what a mark binds to, where binding it elsewhere would say
      something different.
    - Extent — where something begins and ends: what a grouping, a span, or a
      notational construct encloses.
    - Substitution — one character or symbol transcribed as another it
      resembles, including a mark that carries meaning being dropped or added.
    - Polarity — a negation gained or lost.
    - Quantity — any change to a value, its magnitude, its precision, or the
      range something is taken over.
    - Relation — a logical, conditional, or ordering connective exchanged for a
      different one.
    - Position — where content sits inside a structure, when the structure is
      what gives it meaning: a cell's row and column, an item's nesting depth,
      a heading's level, the indentation that places a line inside a block of
      code.
    - Order — content sequenced in a way the page does not support, such as
      material lifted out of a separate region and interleaved with the body.
    - Presence — content the transcription dropped or duplicated, other than
      the page furniture named below.

    Judge by effect rather than by this list: if the transcription asserts
    something the page does not, correct it.

    WHAT NOT TO TOUCH

    - The document's substance. Transcribe the source's own errors faithfully —
      a wrong step, a bad value, a claim that does not follow. You are checking
      the transcription, not the document.
    - Arrangement. Do not reorganise content that already follows the page's
      own order.
    - Numbering and labels. Never renumber or re-letter anything.
    - Notation and terminology. Keep the document's conventions and symbols as
      they are; do not standardise them.
    - Formatting. Markdown structure, math delimiters, emphasis, and whitespace
      that only affects appearance stay exactly as transcribed, even where you
      would write them differently. Whitespace that carries meaning is not
      formatting and belongs to Position above: indentation inside a block of
      code is structure, and a line indented to the wrong depth says something
      the page does not.
    - Page furniture. Running heads, folios, and marginal labels are out of
      scope in both directions — leave them wherever the transcription has
      them, and do not add them where it has none, even if the page shows
      them. A footnote is not furniture. Neither is an entry in a reference
      list. Both are content wherever they sit on the page, and a citation of
      a published work is read character by character like any other
      low-redundancy content: every author, title, year, page range, and
      identifier is checked against the image and kept.
    - Wording. Do not reword anything that matches the image.
    - Boundaries. Content that starts or ends abruptly at the edge of the page
      stays that way — do not complete or trim it.

    Return the full corrected markdown for the page and nothing else. If the
    transcription already matches the image, return it unchanged.
    """

    page_image: dspy.Image = dspy.InputField(
        description='The image of the document page — the ground truth to check the transcription against.'
    )
    transcription: str = dspy.InputField(
        description='The OCR markdown transcription of the page to proofread.'
    )
    corrected: str = dspy.OutputField(
        description='The full corrected markdown transcription of the page, with only meaning-changing transcription errors fixed.'
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
        recorder.record_example(
            'corrector',
            {'page_image': page_image, 'transcription': transcription},
            result,
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

        The correction is taken exactly as returned — the page becomes
        whatever the model produced, unexamined and unaltered.

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
        return {'correction_results': [(segment.index, corrected)]}

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
