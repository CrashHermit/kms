from types import SimpleNamespace

import dspy
from langgraph.types import Send

from kms.core import content, models, module, state


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

    Return only the boolean decision. Do not explain, describe findings, or
    rewrite text.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    original_text: str = dspy.InputField(
        description='The original Mistral transcription for this block only.'
    )
    needs_correction: bool = dspy.OutputField(
        description='True only when a visible correction is needed.'
    )


class BlockReviewer(module.Module):
    signature = BlockReviewSignature
    record_name = 'corrector_block_review'
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
            'original_text': original_text,
        }

    def decode(self, prediction, **inputs) -> bool:
        return bool(prediction.needs_correction)

    async def areview(self, region) -> bool:
        return await self.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=region.block.content or '',
        )


class BlockCorrectionSignature(dspy.Signature):
    r"""
    Correct exactly one OCR block using the one supplied block crop as the
    authority.
    Audit every visible character before writing the answer. The task is image
    fidelity, not mathematical correction, formatting cleanup, or rewriting.

    Use LaTeX for all visible mathematical notation and faithfully represent
    the crop. Preserve the original transcription exactly unless the image
    proves a content error. Make the smallest possible correction for every proven
    mismatch, including subtle signs and notation: minus/plus, equality and
    inequality symbols, `\Rightarrow` versus `\Leftrightarrow`, `\cap` versus
    `\cup`, letters and digits, superscripts, subscripts, radical indices,
    fraction numerators/denominators, parentheses/brackets/braces, punctuation,
    table cells, list items, code tokens, and missing or extra visible words or
    lines. Check the scope and attachment of operators, exponents, radicals,
    fractions, and function arguments character by character.

    Do not infer content outside the crop or copy neighboring blocks. Do not
    solve an exercise, repair a false statement, improve grammar, or normalize
    Markdown/LaTeX. Preserve the supplied Markdown and LaTeX representation
    byte-for-byte everywhere except the smallest spans containing proven visual
    content errors. Never add or remove heading markers, emphasis markers,
    dollar delimiters, braces, or other markup just to make the syntax nicer;
    these are not OCR corrections. If the supplied transcription is faithful,
    return it byte-for-byte unchanged and return an empty changes list.

    Examples:

    - Original: `The forms all mean $P \\Rightarrow Q$:`
      The crop visibly shows `P \\Leftrightarrow Q`.
      Return corrected_text with only `\\Rightarrow` changed to
      `\\Leftrightarrow`, and list that one correction in changes.
    - Original: `Figure: $y' = y$`
      The crop visibly shows `Figure: $y' = -y$`.
      Return the same text with only the missing minus sign inserted.
    - Original: `### Exercises 1.3`
      The crop shows the heading text but cannot show Markdown `###`.
      Return the original unchanged and return `changes: []`.
    - Original: a mathematically false but clearly visible equation.
      Return it unchanged. OCR correction must not solve or fact-check it.

    Return the complete corrected block in Markdown plus a concise list of the
    changes actually made—a concise list of the changes, limited to visual
    content corrections. Do not return
    explanations
    outside the corrected_text and changes fields.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    original_text: str = dspy.InputField(
        description='The original Mistral transcription for this block only.'
    )
    corrected_text: str = dspy.OutputField(
        description='The complete corrected Markdown transcription.'
    )
    changes: list[str] = dspy.OutputField(
        description='A concise list of changes made, or an empty list.'
    )


class BlockCorrector(module.Module):
    signature = BlockCorrectionSignature
    record_name = 'corrector_block'
    use_chain_of_thought = False

    def __init__(self, language_model, recorder=None) -> None:
        super().__init__(language_model, recorder)
        self.reviewer = BlockReviewer(language_model, recorder)

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
            'original_text': original_text,
        }

    def decode(self, prediction, **inputs) -> dict:
        return {
            'corrected_text': prediction.corrected_text,
            'changes': module.as_list(prediction.changes),
        }

    async def acorrect(self, region) -> dict:
        original_text = region.block.content or ''
        if not await self.reviewer.areview(region):
            return {'corrected_text': original_text, 'changes': []}
        return await self.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=original_text,
        )


class BlockCorrectorNode:
    """Applies visual block corrections to canonical document Markdown."""

    def __init__(self, corrector: BlockCorrector) -> None:
        self.corrector = corrector

    def dispatch(self, current_state: state.State) -> list[Send] | str:
        """Sends one correction worker per document with OCR crops."""
        sends = [
            Send('block_corrector_worker', {'document': document})
            for document in current_state.get('documents', [])
            if any(
                node.provenance.get('crop_path')
                for node in document.nodes
            )
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
        return {
            'block_correction_results': [(document.index, document.nodes)]
        }

    def collect(self, current_state: state.State) -> dict:
        """Writes corrected Markdown back onto canonical documents."""
        documents = current_state['documents']
        by_index = dict(current_state.get('block_correction_results', []))
        for document in documents:
            if document.index in by_index:
                document.nodes = by_index[document.index]
        return {'documents': documents}
