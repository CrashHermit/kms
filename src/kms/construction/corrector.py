"""OCR proofreading: fixes transcription errors against the page image."""

import logging

import dspy
from langgraph.types import Send

from kms.core import content, models, module, state
from kms.core.edits import LineEdit, apply_line_edits, number_lines

logger = logging.getLogger(__name__)


class MathSignature(dspy.Signature):
    r"""
    Check the numbered transcription against the page image. Review only
    mathematical and notation fidelity. Return line edits only when the image
    proves the transcription is wrong.

    Check symbols, digits, signs, exponents, subscripts, fractions, radicals,
    grouping, equation extent, inequalities, Greek letters, matrices, tables,
    and mathematical part markers. Preserve the source's mathematical meaning
    even when the source itself is incorrect. Do not format Markdown, add math
    delimiters, simplify expressions, or rewrite surrounding prose.

    Every replacement must preserve the rest of its line exactly. Return an
    empty list when no visual correction is certain. Return only the edits.
    """

    page_image: dspy.Image = dspy.InputField(
        description='The document page image, which is the only authority.'
    )
    lines: str = dspy.InputField(
        description='The OCR transcription as numbered Markdown lines.'
    )
    edits: list[LineEdit] = dspy.OutputField(
        description='Meaning-changing mathematical corrections, by line number.'
    )


class ProseSignature(dspy.Signature):
    r"""
    Check the numbered transcription against the page image. Review only
    ordinary prose fidelity. Return line edits only when the image proves the
    transcription is wrong.

    Check missing or duplicated words, clear character substitutions, broken
    words, and meaning-changing punctuation or capitalization. Preserve the
    source's wording, terminology, claims, and errors. Do not correct
    mathematics, improve grammar or style, change Markdown, or rewrite a line
    that already matches the image.

    Every replacement must preserve the rest of its line exactly. Return an
    empty list when no visual correction is certain. Return only the edits.
    """

    page_image: dspy.Image = dspy.InputField(
        description='The document page image, which is the only authority.'
    )
    lines: str = dspy.InputField(
        description='The OCR transcription as numbered Markdown lines.'
    )
    edits: list[LineEdit] = dspy.OutputField(
        description='Meaning-changing prose corrections, by line number.'
    )


class LayoutSignature(dspy.Signature):
    r"""
    Check the numbered transcription against the page image. Review only
    visual content and placement. Return line edits only when the image proves
    the transcription is wrong.

    Check missing or duplicated content, line order, list or table position,
    indentation, headings, image placeholders, and exercise part markers.
    Preserve every source word and the page's own content. Do not correct
    mathematical or prose characters, normalize Markdown, remove page
    furniture, or fill in missing content by inference.

    Every replacement must preserve the rest of its line exactly. Return an
    empty list when no visual correction is certain. Return only the edits.
    """

    page_image: dspy.Image = dspy.InputField(
        description='The document page image, which is the only authority.'
    )
    lines: str = dspy.InputField(
        description='The OCR transcription as numbered Markdown lines.'
    )
    edits: list[LineEdit] = dspy.OutputField(
        description='Certain visual layout and presence corrections, by line number.'
    )


class _ProposalModule(module.Module):
    signature: type[dspy.Signature]
    record_name: str

    def encode(self, page_image: dspy.Image, transcription: str) -> dict:
        return {
            'page_image': page_image,
            'lines': number_lines(transcription),
        }

    def decode(self, prediction, **inputs) -> list[LineEdit]:
        return module.as_list(prediction.edits)


class MathCorrector(_ProposalModule):
    signature = MathSignature
    record_name = 'corrector_math'


class ProseCorrector(_ProposalModule):
    signature = ProseSignature
    record_name = 'corrector_prose'


class LayoutCorrector(_ProposalModule):
    signature = LayoutSignature
    record_name = 'corrector_layout'


def consolidate_edits(edits: list[LineEdit], line_count: int) -> list[LineEdit]:
    accepted: dict[int, LineEdit] = {}
    for edit in edits:
        if not 1 <= edit.index <= line_count:
            raise RuntimeError(
                f'edit line {edit.index} out of range ({line_count} lines)'
            )
        previous = accepted.get(edit.index)
        if previous is None:
            accepted[edit.index] = edit
        elif previous.replacement != edit.replacement:
            logger.warning(
                'rejecting conflicting correction proposals for line %d',
                edit.index,
            )
    return [accepted[index] for index in sorted(accepted)]


class Corrector:
    def __init__(
        self,
        language_model: dspy.LM,
        recorder=None,
    ) -> None:
        self.math = MathCorrector(language_model, recorder)
        self.prose = ProseCorrector(language_model, recorder)
        self.layout = LayoutCorrector(language_model, recorder)

    async def aforward(self, page_image: dspy.Image, transcription: str) -> str:
        proposals: list[LineEdit] = []
        for specialist in (self.math, self.prose, self.layout):
            proposals.extend(
                await specialist.aforward(
                    page_image=page_image, transcription=transcription
                )
            )
        edits = consolidate_edits(proposals, len(transcription.split('\n')))
        return apply_line_edits(transcription, edits)


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
            page_image=content.load_image(segment.image_path),
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
