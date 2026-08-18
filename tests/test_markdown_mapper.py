import pytest

from kms.construction import block_corrector, markdown_mapper
from kms.core.edits import LineEdit


def test_block_prompt_requires_visual_markdown_and_change_output():
    prompt = block_corrector.BlockCorrectionSignature.__doc__
    assert 'one supplied block crop' in prompt
    assert 'Use LaTeX for all visible' in prompt
    assert 'faithfully represent' in prompt
    assert 'concise list of the changes' in prompt
    assert (
        'corrected_text'
        in block_corrector.BlockCorrectionSignature.output_fields
    )
    assert 'changes' in block_corrector.BlockCorrectionSignature.output_fields
    assert (
        'needs_correction' in block_corrector.BlockReviewSignature.output_fields
    )
    assert block_corrector.BlockReviewer.use_chain_of_thought is False
    assert block_corrector.BlockCorrector.use_chain_of_thought is False


def _correction(original='old', corrected='new'):
    return markdown_mapper.BlockCorrection(
        block_index=1,
        block_type='text',
        original_text=original,
        corrected_text=corrected,
    )


def test_encode_numbers_markdown_and_omits_unchanged_blocks():
    mapper = markdown_mapper.MarkdownMapper
    encoded = mapper.encode(
        mapper.__new__(mapper),
        'first\nsecond',
        [_correction(), _correction('same', 'same')],
    )
    assert encoded['lines'] == '[1] first\n[2] second'
    assert 'old' in encoded['block_corrections']
    assert 'same' not in encoded['block_corrections']


def test_apply_mapped_edits_rejects_structural_operations():
    with pytest.raises(RuntimeError, match='non-replacement'):
        markdown_mapper.apply_mapped_edits(
            'a',
            [LineEdit(index=1, operation='delete', replacement='')],
        )


def test_apply_mapped_edits_uses_shared_line_edit_application():
    result = markdown_mapper.apply_mapped_edits(
        'a\nb',
        [LineEdit(index=2, replacement='B')],
    )
    assert result == 'a\nB'
