from types import SimpleNamespace

import dspy
from langgraph.types import Send

from kms.core import content, models, module, state
from kms.core.edits import LineEdit, apply_line_edits, number_lines


class BlockReviewSignature(dspy.Signature):
    r"""
    Perform a strict, character-level visual proofread of exactly one OCR block.

    Compare the crop against the transcription itself. Return TRUE if even one
    visible content error is present or if visible content is missing, including
    a wrong or missing minus sign, plus sign, equality/inequality, relation
    symbol, quantifier, superscript, subscript, radical index, fraction part,
    delimiter, parenthesis, bracket, brace, letter, digit, punctuation mark,
    table cell, list item, code token, or word. Pay special attention to
    low-redundancy mathematical and STEM notation: `=`, `\neq`, `\leq`,
    `\geq`, `\Rightarrow`, `\Leftrightarrow`, `\cap`, `\cup`, signs,
    exponents, subscripts, and the scope of parentheses and radicals.

    Use this mandatory comparison procedure: (1) read the transcription once;
    (2) inspect the crop again from left to right; (3) compare every word,
    digit, and notation glyph; and (4) return TRUE for any mismatch. Never
    infer a glyph from the surrounding meaning, and never trust a plausible
    sentence or a repeated expression elsewhere on the page. A transcription
    can contain exactly one wrong symbol in otherwise perfect prose. Inspect
    mathematical symbols embedded in ordinary sentences and captions just as
    carefully as standalone equations. In particular, inspect each relation
    and connective directly in the image: `=` vs `\\neq`, `<` vs `\\leq`, `>`
    vs `\\geq`, `\\Rightarrow` vs `\\Leftrightarrow`, `\\cap` vs `\\cup`,
    and every visible plus or minus sign are distinct glyphs.

    Also inspect parentheses, brackets, and braces as individual glyphs:
    `(`, `)`, `[`, `]`, `\{`, `\}`. A missing, extra, or moved parenthesis
    is a visual error even if all other symbols match.

    Example: The crop shows `$f(A_1 \cap A_2)` but the transcription says `$f(A_1) \cap A_2` — the parentheses enclose a different scope. Return TRUE.

    Do not use mathematical plausibility, grammar, or formatting preference to
    override the image. Do not solve, simplify, or correct the author's work.
    Markdown syntax alone is not a correction: do not flag a heading level,
    emphasis marker, dollar delimiter, or LaTeX style merely because another
    representation would be nicer. Return FALSE only when the visible content
    is faithful, or when the crop is too unclear to prove an error.

    Examples:

    - The crop visibly shows `P \\Leftrightarrow Q`, while the transcription
      says `P \\Rightarrow Q`: return TRUE. One connective is wrong.
    - The crop visibly shows `y = -x`, while the transcription says `y = x`:
      return TRUE. A minus sign is missing.
    - The crop and transcription agree, even if the sentence is mathematically
      surprising: return FALSE. Do not repair the author's mathematics.
    - The only difference is that the transcription uses `### Heading` or
      `$x$` markup: return FALSE. Markdown representation is not a visual
      content error.

    `block_type` is provider metadata, not an instruction about how to
    interpret the content. If the provider label is unfamiliar (for example,
    `references`), treat the crop as opaque source content and compare only
    the visible transcription. Do not reclassify, merge, split, omit, or
    rewrite a block because of its label.

    Return FALSE when the crop is too blurry, clipped, or ambiguous to prove
    an error. A TRUE decision only authorizes the editor to inspect the block;
    it does not authorize a speculative correction.

    Return only the boolean decision. Do not explain, describe findings, or
    rewrite text.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    lines: str = dspy.InputField(
        description=(
            'The block transcription as 1-based numbered Markdown lines.'
        )
    )
    needs_correction: bool = dspy.OutputField(
        description='True only when a visible correction is needed.'
    )


class BlockCorrectionRouter(module.Module):
    signature = BlockReviewSignature
    record_name = 'corrector_block_router'
    use_chain_of_thought = False

    def encode(
        self,
        crop_path: str,
        block_type: str,
        original_text: str,
    ) -> dict:
        image = content.load_image(crop_path)
        if image is None:
            raise ValueError(f'block crop does not exist: {crop_path}')
        return {
            'block_crop': image,
            'block_type': block_type,
            'lines': number_lines(original_text),
        }

    def decode(self, prediction, **inputs) -> bool:
        return bool(prediction.needs_correction)

    async def needs_correction(self, region) -> bool:
        return await self.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=region.block.content or '',
        )


class BlockCorrectionSignature(dspy.Signature):
    r"""
    Correct exactly one OCR block using the supplied crop as the authority.
    The crop and numbered block lines are the complete input unit. Return only
    replacement edits for changed local lines; never rewrite the complete
    block in the output.

    Audit every visible character before writing the answer. This is image
    fidelity, not mathematical correction, formatting cleanup, or rewriting.
    Preserve the supplied Markdown and LaTeX representation everywhere except
    the smallest spans containing proven visual content errors. Do not infer
    content outside the crop or copy neighboring blocks. Do not solve an
    exercise, repair a false statement, improve grammar, or normalize markup.

    Each edit must contain a 1-based local line index, operation `replace`, and
    the complete replacement line. Emit no edit for an unchanged line. Never
    use delete, insert_before, or insert_after. Preserve the original line
    structure; only use embedded line breaks when the visible correction
    unambiguously changes it. If the block is faithful, return `[]`.

    Make the smallest possible correction for signs, relation symbols, letters,
    digits, superscripts, subscripts, radical indices, fraction parts,
    parentheses, punctuation, table cells, list items, code tokens, and visible
    missing or extra words. Preserve every other character exactly. Markdown
    markers and delimiters are not visual corrections. A mathematically false
    but clearly visible equation remains unchanged.

    `block_type` is provider metadata, not a semantic editing instruction.
    Unknown labels such as `references` must be preserved as opaque source
    content. Never reclassify the block, merge it with neighboring content,
    or change its boundaries. Require direct visual evidence for every edit;
    distinguish `\\Rightarrow` from `\\Leftrightarrow`, `=` from `\\neq`, and
    every plus/minus sign by the glyph itself rather than by plausibility or
    sentence meaning. If the glyph is not clearly visible, emit no edit.

    Also inspect parentheses, brackets, and braces as individual glyphs:
    `(`, `)`, `[`, `]`, `\{`, `\}`. A missing, extra, or moved parenthesis
    is a visual error even if all other symbols match. Correct the placement
    of parentheses when the crop shows a different grouping, such as
    `f(A) \cap B` versus `f(A \cap B)`.

    Example: The crop shows `$f(A_1 \cap A_2)` but the transcription says `$f(A_1) \cap A_2`. Emit a replacement edit for the full line with corrected parentheses scope.

    Return only the replacement edit list.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    lines: str = dspy.InputField(
        description=(
            'The OCR transcription for this block as 1-based numbered '
            'Markdown lines.'
        )
    )
    edits: list[LineEdit] = dspy.OutputField(
        description=(
            'Replacement-only edits for changed local lines. Empty when the '
            'block is faithful.'
        )
    )


class BlockCorrectionEditor(module.Module):
    signature = BlockCorrectionSignature
    record_name = 'corrector_block_editor'
    use_chain_of_thought = False

    def encode(
        self,
        crop_path: str,
        block_type: str,
        original_text: str,
    ) -> dict:
        image = content.load_image(crop_path)
        if image is None:
            raise ValueError(f'block crop does not exist: {crop_path}')
        return {
            'block_crop': image,
            'block_type': block_type,
            'lines': number_lines(original_text),
        }

    def decode(self, prediction, **inputs) -> list[LineEdit]:
        edits = module.as_list(prediction.edits)
        if any(edit.operation != 'replace' for edit in edits):
            raise RuntimeError(
                'block corrector editor emitted a non-replacement edit'
            )
        return edits


class BlockCorrector:
    def __init__(
        self,
        language_model=None,
        recorder=None,
        router_language_model=None,
        editor_language_model=None,
    ) -> None:
        router_language_model = router_language_model or language_model
        editor_language_model = editor_language_model or language_model
        if router_language_model is None or editor_language_model is None:
            raise ValueError('block corrector language models are required')
        self.router = BlockCorrectionRouter(router_language_model, recorder)
        self.editor = BlockCorrectionEditor(editor_language_model, recorder)

    async def acorrect(self, region) -> dict:
        original_text = region.block.content or ''
        if not await self.router.needs_correction(region):
            return {'corrected_text': original_text, 'edits': []}
        edits = await self.editor.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=original_text,
        )
        return {
            'corrected_text': apply_line_edits(original_text, edits),
            'edits': edits,
        }


class BlockCorrectorNode:
    """Applies visual block corrections to canonical document Markdown."""

    def __init__(self, corrector: BlockCorrector) -> None:
        self.corrector = corrector

    def dispatch(self, current_state: state.State) -> list[Send] | str:
        """Sends one correction worker per document with OCR crops."""
        sends = [
            Send('block_corrector_worker', {'document': document})
            for document in current_state.get('documents', [])
            if any(node.provenance.get('crop_path') for node in document.nodes)
        ]
        return sends or 'block_corrector_collect'

    async def worker(self, current_state: dict) -> dict:
        """Reviews every cropped OCR block in one document."""
        document: models.Document = current_state['document']
        for node in document.nodes:
            crop_path = node.provenance.get('crop_path')
            if not crop_path:
                continue
            result = await self.corrector.acorrect(
                SimpleNamespace(
                    crop_path=crop_path,
                    block=SimpleNamespace(
                        type=node.provenance.get('provider_type', node.type),
                        content=node.content,
                    ),
                )
            )
            node.content = result['corrected_text']
        return {'block_correction_results': [(document.index, document.nodes)]}

    def collect(self, current_state: state.State) -> dict:
        """Writes corrected Markdown back onto canonical documents."""
        documents = current_state['documents']
        by_index = dict(current_state.get('block_correction_results', []))
        for document in documents:
            if document.index in by_index:
                document.nodes = by_index[document.index]
        return {'documents': documents}
