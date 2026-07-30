r"""
Formatting pass over each corrected page.

The corrector answers one question — does the transcription say what the page
says — and is forbidden from touching presentation, because its authority is the
page image and the image cannot settle how markdown ought to be written. That
leaves presentation to this stage, which answers the complementary question:
is the page written down the way every source is written down?

The two passes are exact inverses, and that is the point of splitting them: the
corrector may change meaning-bearing content and may not reformat; the formatter
may reformat and may not change meaning. Neither has to trade one objective off
against the other, and each can later be optimised against a metric that suits
it — fidelity to an image for the corrector, conformance to stated rules here.

Text only, no page image: standardising markup needs the markdown and the rules,
not the page, so this runs on the cheap text LM rather than the vision model.
It sits *after* the corrector for a hard reason — anything that deliberately
makes the text diverge from the image (a delimiter the page does not show, a
list marker the author did not use) must happen after the pass whose contract is
that text and image agree, or the corrector will faithfully undo it.

It closes a gap the corrector's split left open: with delimiter normalization
removed from the corrector, nothing was converting `\( … \)` / `\[ … \]` to the
dollar convention the extractor's prompt and every downstream stage assume.

**Unguarded**, like the corrector: whatever the model returns is what the page
becomes. No divergence check, no post-processing — the pipeline's passes are
bare LLM calls and correctness rests on the prompt.

Delimiter conversion is stated as its own required section rather than as one
bullet among equals, because a flat rule list was measurably applied only in
part: on a code-heavy real page the pass reliably fenced the code, normalised
heading levels, and left all fourteen `\( … \)` delimiters — including ones on a
heading line it had just edited — the same way on every repeat. Which rules
fired depended on what else the page had wrong with it, so the rule this stage
was added for needed to stop competing for attention with markup housekeeping.

Two deliberate omissions, both of which were on the table and both of which
would break things at this position in the pipeline:

- **Reordering.** Document order is load-bearing downstream: `flatten_segments`
  assigns node ids by position, the pedagogical component finder cuts
  *contiguous* spans, and the procedural layer threads `:FIRST`/`:THEN` in
  stream order. A resequencing pass here would corrupt all three silently.
- **Renumbering and re-lettering.** An exercise's `(b)` is a referent — the
  prose says "by part (b)" — and those references are document-scope while this
  stage, like the corrector, sees one page. It would relabel the item and leave
  the reference dangling with nothing downstream able to notice.

Both are reachable by editing the prompt if the pipeline later grows a
document-scope stage that can do them safely.
"""

import asyncio
import logging

import dspy
from langgraph.types import Send

from kms.core import llm, models, state

logger = logging.getLogger(__name__)


class Signature(dspy.Signature):
    r"""
    You are a meticulous formatter of document transcriptions. You are given one
    page of a document as markdown. Return the same page with its formatting
    standardised.

    Change how the content is written down, never what it says. Every word,
    number, symbol, and mathematical expression must survive unchanged — only
    the markup around them may change. Where a change would alter meaning, or
    where you cannot make it without guessing at meaning, leave the text as it
    is.

    REQUIRED — MATH DELIMITERS

    This is the one change you must always make, and the reason this pass
    exists. Work through the page and convert every occurrence, wherever math
    appears — in prose, in a heading, in a list item, in a table cell:

    - `\( … \)` becomes `$ … $`
    - `\[ … \]` becomes `$$ … $$`
    - a display equation left bare — a standalone equation line, or an `array`
      / `aligned` / `cases` / `equation` environment — is wrapped in `$$ … $$`

    Change the delimiters only, never the expression between them. Convert all
    of them, not the first few, and do this even when the page has other things
    wrong with it.

    ALSO STANDARDISE

    - Headings. Mark every heading with `#`s, one level per structural level,
      deepening consistently down the page. A heading written as a line of text
      underlined by `===` or `---` on the next line is a heading: replace both
      lines with a single `#`-marked one. Do not invent a heading, remove one,
      or promote a line that is not one.
    - Lists. `-` for bullets and `1.` numbering for ordered lists, with nesting
      shown by indentation. Keep every item, its position, and any label the
      document gives it.
    - Emphasis. Write italics as `*italic*` and bold as `**bold**`. Normalise
      the emphasis that is there; do not add emphasis to text that has none.
    - Tables. Pipe tables with a header separator row, one row per line.
    - Blank lines. One blank line between top-level blocks, none inside a
      block.

    WHAT NOT TO TOUCH

    - Figure placeholders. A placeholder of the form `![N]()` must survive
      exactly, digit for digit, with its empty parentheses. It is a positional
      reference resolved later against that page's extracted figures; rewriting
      it, renumbering it, giving it a caption, or filling in a path loses the
      figure.
    - Code and verbatim content. Leave fenced blocks and inline code alone,
      including their indentation and internal spacing — there, whitespace is
      structure, not presentation.
    - Mathematical content. Only the delimiters around an expression may
      change. Never rewrite the expression, and never convert notation to a
      form you prefer.
    - Notation, terminology, and spelling. Keep the document's own conventions
      and symbols; standardise the markup, not the author.
    - Numbering and labels. Leave every identifier the document uses — section
      and theorem numbers, exercise numbers, part letters — exactly as written.
      They are referred to elsewhere by name.
    - Order. Return the content in the order it arrives.
    - Page furniture. Leave running heads, folios, and marginal labels where
      they are; neither delete them nor add ones that are absent. A footnote
      is not furniture, and neither is an entry in a reference list: both are
      content, including the block of them that may sit at the foot of the
      page. Keep every citation, with its authors, title, year, page range,
      and identifiers exactly as written — a reference is a run of proper
      nouns and numbers where a "tidied" character is a changed fact.
    - Content. Add nothing and remove nothing, including anything that starts
      or ends abruptly at the edge of the page.

    Return the full formatted markdown for the page and nothing else. If it
    already follows the conventions above, return it unchanged.
    """

    markdown: str = dspy.InputField(
        description='The corrected markdown of one document page, to standardise.'
    )
    formatted: str = dspy.OutputField(
        description='The full page markdown with its formatting standardised and its content unchanged.'
    )


class Formatter(dspy.Module):
    """Standardises one page's markdown formatting.

    Args:
        language_model: The LM to run on. Defaults to ``llm.text_lm()``.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.formatter = dspy.Predict(Signature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(self, markdown: str) -> str:
        """Format one page.

        Args:
            markdown: The page's corrected markdown.

        Returns:
            The page's markdown with standardised formatting.
        """
        result = await self.formatter.acall(markdown=markdown)
        formatted = result.formatted or ''
        logger.debug(
            'format: %d chars in, %d chars out', len(markdown), len(formatted)
        )
        return formatted

    def forward(self, markdown: str) -> str:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(markdown))


# --- LangGraph node: standardise each corrected page's formatting ---


class FormatterNode:
    """Fans out per-page formatters and collects the standardised text.

    Args:
        module: The formatting module. Created fresh if None.
    """

    def __init__(self, module: Formatter | None = None) -> None:
        self.module = module or Formatter()

    def dispatch(self, state: state.State) -> list[Send] | str:
        """Fan out one worker per page with content.

        No page image is needed here, so unlike the corrector a segment
        qualifies on content alone.

        Args:
            state: The pipeline state, holding the segment backbone.

        Returns:
            One Send per qualifying segment, or the collect step's name when
            none qualify (the stage is then a no-op).
        """
        segments = state.get('segments', [])
        sends = [
            Send('formatter_worker', {'segment': segment})
            for segment in segments
            if segment.content
        ]
        return sends or 'formatter_collect'

    async def worker(self, state: dict) -> dict:
        """Standardise one page's formatting.

        The result is taken exactly as returned — the page becomes whatever the
        model produced, unexamined and unaltered.

        Args:
            state: The worker payload, holding its ``segment``.

        Returns:
            The page's ``format_results`` entry.
        """
        segment: models.Segment = state['segment']
        formatted = await self.module.aforward(markdown=segment.content)
        return {'format_results': [(segment.index, formatted)]}

    def collect(self, state: state.State) -> dict:
        """Write each formatted page back into its segment.

        Segments that were not dispatched keep their content untouched.

        Args:
            state: The pipeline state, holding the formatting results.

        Returns:
            The updated segment backbone.
        """
        results = state.get('format_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'content'
        )
        logger.info('formatter: %d page(s) formatted', len(results))
        return {'segments': segments}
