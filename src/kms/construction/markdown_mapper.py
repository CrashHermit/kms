import json

import dspy
from pydantic import BaseModel

from kms.core import edits as line_edits
from kms.core import module


class BlockCorrection(BaseModel):
    block_index: int
    block_type: str
    original_text: str
    corrected_text: str


class MarkdownMappingSignature(dspy.Signature):
    r"""
    Map visually corrected OCR blocks back onto the numbered Markdown page.

    The numbered Markdown is the document to edit. BLOCK CORRECTIONS contains
    original and image-corrected text for selected Mistral blocks. Compare the
    original and corrected text, locate only the corresponding Markdown lines,
    and emit replacement edits for lines whose visible content changed.

    Preserve every Markdown line that is not required to apply a block
    correction. A replacement must contain the complete new line text, not a
    fragment. Preserve Markdown delimiters, image placeholders, blank lines,
    line order, and all source content. A corrected block may span several
    Markdown lines; emit one replacement per changed line, or use one
    replacement containing line breaks when the block's line boundaries are
    unambiguous.

    This is a replacement-only mapping task. Never emit insert_before,
    insert_after, or delete. Do not add content that is absent from a corrected
    block, solve or explain an exercise, normalize Markdown, fix grammar, or
    infer a correction that is not present in BLOCK CORRECTIONS. If no mapped
    line changes, return an empty list. Return only the edits.
    """

    lines: str = dspy.InputField(
        description='The complete page Markdown with 1-based line numbers.'
    )
    block_corrections: str = dspy.InputField(
        description=(
            'JSON array of corrected Mistral blocks, each with block_index, '
            'block_type, original_text, and corrected_text.'
        )
    )
    edits: list[line_edits.LineEdit] = dspy.OutputField(
        description='Replacement-only edits that apply the block corrections.'
    )


class MarkdownMapper(module.Module):
    signature = MarkdownMappingSignature
    record_name = 'corrector_markdown_mapper'

    def encode(
        self,
        markdown: str,
        corrections: list[BlockCorrection],
    ) -> dict:
        changed = [
            correction.model_dump(mode='json')
            for correction in corrections
            if correction.original_text != correction.corrected_text
        ]
        return {
            'lines': line_edits.number_lines(markdown),
            'block_corrections': json.dumps(
                changed, ensure_ascii=False, indent=2
            ),
        }

    def decode(self, prediction, **inputs) -> list[line_edits.LineEdit]:
        return module.as_list(prediction.edits)


def validate_replacement_edits(
    edits: list[line_edits.LineEdit],
) -> list[line_edits.LineEdit]:
    invalid = [edit for edit in edits if edit.operation != 'replace']
    if invalid:
        raise RuntimeError('markdown mapper emitted a non-replacement edit')
    return edits


def apply_mapped_edits(
    markdown: str,
    edits: list[line_edits.LineEdit],
) -> str:
    return line_edits.apply_line_edits(
        markdown,
        validate_replacement_edits(edits),
    )


async def map_and_apply(
    mapper: MarkdownMapper,
    markdown: str,
    corrections: list[BlockCorrection],
) -> str:
    edits = await mapper.aforward(markdown=markdown, corrections=corrections)
    return apply_mapped_edits(markdown, edits)
