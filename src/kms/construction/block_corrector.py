import logging
from types import SimpleNamespace

import dspy
from langgraph.types import Send

from kms.core import edits, images, models, module, state
from kms.core.edits import LineReplacement

logger = logging.getLogger(__name__)


class BlockReviewSignature(dspy.Signature):
    r"""
    Decide whether this OCR block contains a visibly proven transcription
    error. This is a visual comparison task, not a math, grammar, or formatting
    task.

    DEFAULT ANSWER: FALSE.

    Return TRUE only when the crop directly proves a specific mismatch between
    the visible content and the transcription and at least one supplied line
    can be named as the mismatched line. If no supplied line can be identified
    with confidence, return FALSE. You must be able to identify the mismatched
    character, word, digit, punctuation mark, or mathematical glyph. If the
    crop is small, blurry, clipped, compressed, ambiguous, or merely
    suspicious, return FALSE. Never guess.
    Compare the crop and transcription character by character from left to
    right. Check words, digits, signs, relation symbols, quantifiers,
    exponents, subscripts, fraction parts, delimiters, parentheses, brackets,
    braces, table cells, and code tokens. Treat `=`, `\neq`, `<`, `\leq`, `>`,
    `\geq`, `\Rightarrow`, `\Leftrightarrow`, `\cap`, `\cup`, `+`, and `-` as
    distinct glyphs. Do not infer a character from mathematical plausibility,
    grammar, repetition, or expected wording.

    Faithful examples return FALSE:
    - The crop and transcription agree, even if the mathematics is surprising.
    - A one-line exercise such as `b) Find an example such that |x_n|
      converges and x_n diverges.` is faithful.
    - The only difference is Markdown or LaTeX presentation.
    - You cannot identify the exact visible mismatched character.

    Mismatch examples return TRUE:
    - The crop visibly shows `y = -x`, while the transcription says `y = x`.
    - The crop visibly shows `P \Leftrightarrow Q`, while the transcription
      says `P \Rightarrow Q`.

    `block_type` is provider metadata, not a correction instruction. Treat
    unfamiliar labels as opaque source content. Do not reclassify, merge, split,
    omit, solve, simplify, or rewrite the block.

    Return only the boolean decision. Do not explain your decision.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    lines: list[models.LineInput] = dspy.InputField(
        description=(
            'Ordered source lines with explicit one-based indexes. The index '
            'is metadata; numbers in text are source content. A one-line '
            'block contains only index 1.'
        )
    )
    needs_correction: bool = dspy.OutputField(
        description='True only when a visible correction is needed.'
    )


class BlockCorrectionLocationSignature(dspy.Signature):
    r"""
    Identify which supplied OCR lines contain a proven visual transcription
    error in the crop. Do not rewrite text and do not judge mathematical
    correctness, formatting, or plausibility.

    Return every supplied line with a directly visible mismatch and no other
    line. Indexes are explicit one-based coordinates; numbers inside text are
    source content. For one input line, only index 1 is valid.

    Faithful example:
    Input: [{"index": 1, "text": "y = x"}]
    Output: []

    Error example:
    Input: [{"index": 1, "text": "y = x"}]
    Output: [{"index": 1}]

    Invalid example:
    Input: [{"index": 1, "text": "one line"}]
    Output: [{"index": 2}]
    This is invalid because index 2 is not present in the input.

    Inspect the crop directly. Do not infer missing content, solve equations,
    repair false mathematics, or select a line merely because its text is
    surprising. Return only the selected supplied line indexes.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    lines: list[models.LineInput] = dspy.InputField(
        description=(
            'Ordered OCR lines with explicit one-based indexes. A one-line '
            'block has exactly one valid index: 1.'
        )
    )
    locations: list[models.LineSelection] = dspy.OutputField(
        description=(
            'Every supplied one-based line index with a proven visible OCR '
            'mismatch. Empty when no line is sufficiently certain.'
        )
    )


class BlockCorrectionRouter(module.Module):
    signature = BlockReviewSignature
    record_name = 'corrector_block_router'

    def encode(
        self,
        crop_path: str,
        block_type: str,
        original_text: str,
    ) -> dict:
        image = images.load_image(crop_path)
        if image is None:
            raise ValueError(f'block crop does not exist: {crop_path}')
        return {
            'block_crop': image,
            'block_type': block_type,
            'lines': edits.numbered_lines(original_text),
        }

    def decode(self, prediction, **inputs) -> bool:
        """Returns the validated correction-routing decision."""
        return module.require_bool(
            prediction.needs_correction, 'needs_correction'
        )

    async def needs_correction(self, region) -> bool:
        return await self.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=region.block.content or '',
        )


class BlockCorrectionLocator(module.Module):
    """Locates visually incorrect source lines before editing."""

    signature = BlockCorrectionLocationSignature
    record_name = 'corrector_block_locator'

    def encode(
        self,
        crop_path: str,
        block_type: str,
        original_text: str,
    ) -> dict:
        image = images.load_image(crop_path)
        if image is None:
            raise ValueError(f'block crop does not exist: {crop_path}')
        return {
            'block_crop': image,
            'block_type': block_type,
            'lines': edits.numbered_lines(original_text),
        }

    def decode(self, prediction, **inputs) -> list[models.LineSelection]:
        locations = module.as_list(prediction.locations)
        original_text = inputs.get('original_text', '')
        input_lines = edits.numbered_lines(original_text)
        indexes = [location.index for location in locations]
        if len(indexes) != len(set(indexes)):
            raise ValueError(
                f'block locator returned duplicate line indices: {indexes}; '
                f'input={input_lines}'
            )
        invalid = [
            index for index in indexes if not 1 <= index <= len(input_lines)
        ]
        if invalid:
            raise ValueError(
                f'block locator returned out-of-range line indices: {invalid}; '
                f'valid range=1..{len(input_lines)}; input={input_lines}'
            )
        return locations

    async def locate(self, region) -> list[models.LineSelection]:
        return await self.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=region.block.content or '',
        )


class BlockCorrectionSignature(dspy.Signature):
    r"""
    Correct exactly one OCR block using the supplied crop as the authority.
    The crop and structured source-line records are the complete input unit.
    Return only replacement edits for changed local lines; never rewrite the
    complete block in the output.

    Audit every visible character before writing the answer. This is image
    fidelity, not mathematical correction, formatting cleanup, or rewriting.
    Preserve the supplied Markdown and LaTeX representation everywhere except
    the smallest spans containing proven visual content errors. Do not infer
    content outside the crop or copy neighboring blocks. Do not solve an
    exercise, repair a false statement, or normalize markup.

    The locator may be uncertain even after a positive router decision. Use
    only indexes from allowed_indexes. An empty replacement list is a safe
    no-op when no edit can be proven; never emit a speculative correction.

    Each replacement contains the explicit one-based index of one supplied
    input line and its complete replacement text. Numbers inside line text are
    source content, not coordinates. For a one-line input, index 1 is valid
    and index 2 is invalid.

    Positive example:
    Input: {"allowed_indexes": [1], "lines": [{"index": 1, "text": "y = x"}]}
    Output: [{"index": 1, "replacement": "y = -x"}]

    Conservative no-op example:
    Input: {"allowed_indexes": [1], "lines": [{"index": 1, "text": "unclear"}]}
    Output: []
    Return no edit when the crop does not prove the correction.

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

    Symbol-fidelity examples:
    - If the crop visibly shows `P \Leftrightarrow Q` but the OCR line says
      `P \Rightarrow Q`, replace only that source line with the
      `\Leftrightarrow` form.
    - If the crop visibly shows `(P \wedge Q)` but the OCR line says
      `(P \vee Q)`, replace only that source line with the `\wedge` form.
    - If the crop visibly shows `x \geq 0` but the OCR line says `x \ge 0`,
      preserve the crop's exact `\geq` command; equivalent LaTeX is not a
      valid reason to normalize the source.
    - If the crop visibly shows `a \neq b` but the OCR line says `a = b`,
      replace the relation symbol with `\neq`; do not infer other changes.
    In every example, preserve all surrounding Markdown, LaTeX delimiters,
    whitespace, and line structure exactly, and return no unchanged lines.

    Return only the replacement edit list.
    """

    block_crop: dspy.Image = dspy.InputField(
        description='The cropped page region for this OCR block.'
    )
    block_type: str = dspy.InputField(
        description='The Mistral block type, such as text or equation.'
    )
    lines: list[models.LineInput] = dspy.InputField(
        description=(
            'Ordered OCR lines with explicit one-based indexes. A one-line '
            'block has exactly one valid index: 1.'
        )
    )
    allowed_indexes: list[int] = dspy.InputField(
        description=(
            'The complete set of one-based line indexes selected by the '
            'locator. Every emitted replacement must use one of these.'
        )
    )
    edits: list[LineReplacement] = dspy.OutputField(
        description=(
            'Replacement-only edits addressing supplied one-based line '
            'indexes. Empty when the block is faithful.'
        )
    )


class BlockCorrectionEditor(module.Module):
    signature = BlockCorrectionSignature
    record_name = 'corrector_block_editor'

    def encode(
        self,
        crop_path: str,
        block_type: str,
        original_text: str,
        allowed_indexes: list[int],
    ) -> dict:
        image = images.load_image(crop_path)
        if image is None:
            raise ValueError(f'block crop does not exist: {crop_path}')
        return {
            'block_crop': image,
            'block_type': block_type,
            'lines': edits.numbered_lines(original_text),
            'allowed_indexes': allowed_indexes,
        }

    def decode(self, prediction, **inputs) -> list[edits.LineReplacement]:
        replacements = module.as_list(prediction.edits)
        original_text = inputs.get('original_text', '')
        input_lines = edits.numbered_lines(original_text)
        allowed_indexes = inputs.get('allowed_indexes', [])
        indexes = [replacement.index for replacement in replacements]
        if len(indexes) != len(set(indexes)):
            raise ValueError(
                'block editor returned duplicate line indices: '
                f'{indexes}; input={input_lines}'
            )
        invalid = [
            index for index in indexes if not 1 <= index <= len(input_lines)
        ]
        unauthorized = [
            index for index in indexes if index not in allowed_indexes
        ]
        if invalid or unauthorized:
            raise ValueError(
                'block editor returned invalid line indices: '
                f'indexes={indexes}; invalid={invalid}; '
                f'allowed_indexes={allowed_indexes}; '
                f'valid range=1..{len(input_lines)}; input={input_lines}'
            )
        return replacements


class BlockCorrector:
    def __init__(
        self,
        language_model=None,
        recorder=None,
        router_language_model=None,
        locator_language_model=None,
        editor_language_model=None,
    ) -> None:
        router_language_model = router_language_model or language_model
        locator_language_model = locator_language_model or router_language_model
        editor_language_model = editor_language_model or language_model
        if (
            router_language_model is None
            or locator_language_model is None
            or editor_language_model is None
        ):
            raise ValueError('block corrector language models are required')
        self.router = BlockCorrectionRouter(router_language_model, recorder)
        self.locator = BlockCorrectionLocator(locator_language_model, recorder)
        self.editor = BlockCorrectionEditor(editor_language_model, recorder)

    async def acorrect(self, region) -> dict:
        original_text = region.block.content or ''
        if not await self.router.needs_correction(region):
            return {'corrected_text': original_text, 'edits': []}
        locations = await self.locator.locate(region)
        if not locations:
            logger.warning(
                'block correction disagreement: router=true, locator=0'
            )
            return {'corrected_text': original_text, 'edits': []}
        allowed_indexes = [location.index for location in locations]
        corrections = await self.editor.aforward(
            crop_path=region.crop_path,
            block_type=region.block.type,
            original_text=original_text,
            allowed_indexes=allowed_indexes,
        )
        if not corrections:
            logger.warning(
                'block correction no-op: router=true, locator_count=%d, '
                'editor=0',
                len(locations),
            )
        return {
            'corrected_text': edits.apply_line_replacements(
                original_text, corrections
            ),
            'edits': corrections,
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
