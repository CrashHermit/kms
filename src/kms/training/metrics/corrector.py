"""Corrector-stage metric — judges a proofread page against its source image.

Uses a VLM judge (``corrector_judge_lm``) that compares the candidate
correction against the actual page image. The reference correction is a
trusted *guide* to where to look; the image is the only ground truth.

Set ``KMS_TRACE_STRIP_IMAGES=0`` during trace collection to keep page images
in traces, otherwise the judge receives ``'<image>'`` placeholders and cannot
verify against the image.
"""

from __future__ import annotations

import dspy

from kms.core import llm
from kms.training.metrics._helpers import _judge


# ============================================================================
# Error model
# ============================================================================


class Error(dspy.Signature):
    """One discrete error found in a correction candidate."""

    type: str = dspy.OutputField(
        desc='One of: unfixed_error, introduced_error, '
             'delimiter_unfixed, runaway_rewrite'
    )
    location: str = dspy.OutputField(
        desc='A short quoted snippet from the candidate that shows the error '
             '(or "N/A" for runaway_rewrite).'
    )
    note: str = dspy.OutputField(
        desc='One sentence describing what is wrong and what the page image '
             'actually shows at that location.'
    )


# ============================================================================
# Judge signature
# ============================================================================


class Judge(dspy.Signature):
    r"""You are a meticulous proofreading judge. You have FOUR things:

    1. A PAGE IMAGE — the only ground truth. Every judgment must ultimately
       trace back to what the image shows.
    2. The original OCR TRANSCRIPTION — what the image was transcribed as.
    3. A REFERENCE correction — a trusted example of what a good correction
       of this page looks like.
    4. A CANDIDATE correction — the output you are evaluating.

    GROUND TRUTH IS THE IMAGE, NOT THE REFERENCE. The reference is a GUIDE:
    it shows you what errors were present in the OCR and how they should be
    fixed. But the reference is itself just one valid markdown rendering of
    the page — it may format things differently from the candidate in ways
    that are BOTH acceptable. Two correct transcriptions of the same page
    can differ in layout, whitespace, markdown structure, and still both be
    faithful to the image.

    WHAT COUNTS AS AN ERROR — only differences that make the candidate LESS
    faithful to what the page image actually says:

    unfixed_error — The OCR had a mistake (visible when you check the image).
      The reference fixed it. The candidate LEFT IT UNFIXED — the wrong text
      from the original OCR is still there. The image clearly shows something
      different from what the candidate says.

    introduced_error — The candidate CHANGED text that was already correct
      (matched the image) in the original OCR. The candidate introduced noise
      where there was none before. Check: look at the image at that spot. If
      the original transcription said what the image says, but the candidate
      changed it to something else, that is an introduced error.

    delimiter_unfixed — The candidate left ``\( ... \)`` or
      ``\[ ... \]`` delimiters unconverted, or left a display equation
      undelimited that should be wrapped in ``$$ ... $$``. This is purely
      structural and easy to spot — flag each occurrence.

    runaway_rewrite — The candidate rewrote large sections of text when the
      original was already faithful to the image. It used different wording,
      restructured paragraphs, or substantially reformatted the page. This
      subsumes all other errors in the affected region — flag ONLY this one
      error, not individual sub-errors within the rewritten stretch.

    WHAT IS NOT AN ERROR — do NOT flag these:
    - Different markdown formatting of the same content (e.g. one candidate
      uses ``**bold**`` where the reference uses ``__bold__``, or different
      line-wrapping of the same paragraph).
    - Different but equally-valid layout decisions (e.g. the reference put a
      figure placeholder on its own line while the candidate kept it inline —
      both are fine as long as the content matches the image).
    - Trailing whitespace, equivalent spacing, or other cosmetic differences.
    - ``\(`` \u2192 ``$`` and ``\[`` \u2192 ``$$`` delimiter conversions —
      these are EXPECTED and correct, never flag them.

    HOW TO JUDGE — for every difference you notice between the reference and
    the candidate:
    1. Look at the page image at that spot.
    2. Ask: does the candidate faithfully represent what the image shows?
    3. If yes — not an error, even if it differs from the reference.
    4. If no — flag it as the appropriate error type above.

    Be exhaustive. List EVERY error you find. If the candidate is faithful
    to the image at every location, return an empty error list — even if it
    differs from the reference in formatting."""

    page_image: dspy.Image = dspy.InputField(
        desc='The textbook page image — THE ground truth. Verify every '
             'disputed location against this image.'
    )
    inputs: str = dspy.InputField(
        desc='The original OCR transcription '
             '(keyed as "transcription: ...").'
    )
    expected: str = dspy.InputField(
        desc='The reference corrected transcription — a trusted GUIDE to '
             'what errors were present and how to fix them. Not the ground '
             'truth.'
    )
    predicted: str = dspy.InputField(
        desc='The candidate corrected transcription — the output to evaluate.'
    )
    errors: list[Error] = dspy.OutputField(
        desc='Every discrete error where the candidate disagrees with the '
             'page image. Empty list if faithful.'
    )
    feedback: str = dspy.OutputField(
        desc='One-sentence summary of what errors were found, or "no errors" '
             'if the candidate is faithful to the image.'
    )


# ============================================================================
# Error weights
# ============================================================================

ERROR_WEIGHTS: dict[str, float] = {
    'unfixed_error': 2.0,
    'introduced_error': 2.0,
    'delimiter_unfixed': 0.5,
    'runaway_rewrite': float('inf'),
}
"""Weight of each error type. A runaway rewrite forces score 0.0."""


# ============================================================================
# Metric
# ============================================================================


def corrector_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: object | None = None,
) -> float | bool:
    """Judge the corrector's output against the page image and reference.

    Uses a discrete-error judge: the VLM lists specific errors
    (unfixed_error, introduced_error, delimiter_unfixed, runaway_rewrite),
    and a deterministic score is derived from the weighted error count. The
    page image is the ground truth; the reference is a trusted guide to
    where to look.

    This is a VLM metric — it uses ``corrector_judge_lm`` (MiMo-V2.5) so
    the judge can compare the candidate correction against the actual page
    image. Requires ``KMS_TRACE_STRIP_IMAGES=0`` during trace collection,
    otherwise the page image is ``'<image>'`` and the judge cannot see it.
    """
    page_image = _coerce_image(example, 'page_image')
    return _judge(
        example,
        prediction,
        input_keys=['transcription'],
        expected_keys=['corrected'],
        predicted_key='corrected',
        judge_signature=Judge,
        judge_lm=llm.corrector_judge_lm,
        error_weights=ERROR_WEIGHTS,
        trace=trace,
        page_image=page_image,
    )


def _coerce_image(
    example: dspy.Example, key: str
) -> dspy.Image:
    """Turn a traced image value back into a ``dspy.Image``.

    Traces store images as base64 data-URL strings (when stripping is off);
    the VLM judge needs a ``dspy.Image``. If the value is already a
    ``dspy.Image`` or the placeholder ``'<image>'``, return it as-is — the
    VLM will see the placeholder text instead of the real image.
    """
    value = getattr(example, key, None)
    if isinstance(value, dspy.Image):
        return value
    if isinstance(value, str) and value.startswith('data:image/'):
        return dspy.Image(url=value)
    # Placeholder or missing — return an empty image so the judge still runs
    # but will likely score low (no image to check against).
    return dspy.Image(url='')
