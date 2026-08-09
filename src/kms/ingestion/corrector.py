import asyncio
import base64
import logging
from pathlib import Path

import dspy
from langgraph.types import Send

from kms.core import models, recording, state

logger = logging.getLogger(__name__)


def _load_dspy_image(path: str | None) -> dspy.Image | None:
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

    SUB-PART MARKERS

    A marker that letters an exercise's parts — `ⓐ`, `ⓑ`, `ⓒ`, `(a)`, `a)` —
    is content, and OCR misreads it often, because it is a glyph rather than a
    letter: `ⓐ` comes back as `$\odot$`, as `©`, as `a`, or as nothing at all.

    Restore the marker the image shows, spelled with the glyph the image
    prints. This licenses nothing beyond the marker itself: the words, the
    notation, and the markup around it stay exactly as transcribed. Restoring
    `ⓐ` in "ⓐ m = 3" does not also mean writing `m = 3` as `$m = 3$` — math
    delimiters are Formatting, they belong to a later pass, and adding them
    here is a change this pass is forbidden to make even when the page image
    shows the mathematics typeset.

    Three ways to get the marker wrong, all of them seen:

    - DELETING the markers. "round to the nearest ⓐ hundred ⓑ thousand ⓒ ten
      thousand" is three sub-parts; transcribed as "the nearest hundred,
      thousand, ten thousand" it is one instruction and the exercise has
      silently lost two of its questions. Never resolve a misread marker by
      dropping it or by replacing it with punctuation. This pass does not
      delete.
    - RESPELLING them. If the page prints `ⓐ`, write `ⓐ` — not `(a)`, not
      `a)`, not `**a**`. Choosing one house form across a book is a later
      pass's job and it works from what you leave; guessing here just hides
      what the page did.
    - Letting one page disagree with itself. The same marker misread twice on
      one page gets the same answer both times.

    The letter itself is an identifier — the prose says "by part (b)" — so it
    is never renumbered or re-lettered, only restored.

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
    - Punctuation that changes no meaning, in BOTH directions. A doubled comma
      in a lead-in ("In the following exercises,, simplify.") asserts nothing
      the page denies, so leave it; equally, do not introduce one the
      transcription lacks. Punctuation that does change meaning — a decimal
      point, a negation, a delimiter inside notation — is Quantity or
      Substitution above and is corrected like anything else.
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
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.proofreader = dspy.Predict(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, page_image: dspy.Image, transcription: str) -> str:
        result = await self.proofreader.acall(
            page_image=page_image, transcription=transcription
        )
        if self._recorder:
            self._recorder.record(
                'corrector',
                {'page_image': page_image, 'transcription': transcription},
                result,
            )
        corrected = result.corrected
        logger.debug(
            'proofread: %d chars in, %d chars out',
            len(transcription),
            len(corrected),
        )
        return corrected

    def forward(self, page_image: dspy.Image, transcription: str) -> str:
        return asyncio.run(self.aforward(page_image, transcription))


class CorrectorNode:
    def __init__(self, module: Corrector) -> None:
        self.module = module

    def dispatch(self, state: state.State) -> list[Send] | str:
        segments = state.get('segments', [])
        sends = [
            Send('corrector_worker', {'segment': segment})
            for segment in segments
            if segment.content and segment.image_path
        ]
        return sends or 'corrector_collect'

    async def worker(self, state: dict) -> dict:
        segment: models.Segment = state['segment']
        corrected = await self.module.aforward(
            page_image=_load_dspy_image(segment.image_path),
            transcription=segment.content,
        )
        return {'correction_results': [(segment.index, corrected)]}

    def collect(self, state: state.State) -> dict:
        results = state.get('correction_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'content'
        )
        logger.info('corrector: %d page(s) proofread', len(results))
        return {'segments': segments}

