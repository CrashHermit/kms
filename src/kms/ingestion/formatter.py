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

The same gap has a wider mouth than delimiter conversion covers. The OCR
front-end's markup varies by page: an end-to-end run came back with page 0
carrying 21 delimited spans and pages 1 and 2 carrying none at all, their
mathematics transcribed as plain text with Unicode glyphs — `x⁴`, `3ˣ`,
`9x + 7 when x = 3`. Converting delimiters cannot help a page that has none,
so those pages reached the equation extractor, whose contract is LaTeX *with*
its delimiters, with nothing it could recognise: it found five equations on
the delimited page and none on the other two. Wrapping undelimited
mathematics, and writing Unicode notation as LaTeX, are therefore required
work here for the same reason delimiter conversion is — this is the one pass
positioned to do it, and the corrector cannot, since restoring a delimiter
the page image does not show is precisely the divergence its contract forbids.

Wrapping is the one rule here that can destroy information rather than merely
fail to add it: a bare quantity in a drill exercise looks much like an
expression, and an item number swallowed into a `$ … $` span is an identifier
the rest of the book cites and nothing downstream can restore. The prompt
therefore states the test, lists what is never wrapped, and makes the
tie-break explicit — when in doubt, leave it bare.

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

from kms.core import models, state
from kms.core.recorder import Recorder

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
    - consecutive display-math blocks that are halves of one equation are joined
      into a single `$$ … $$` block. Judge the second block: if it opens with a
      relational operator (`=`, `<`, `>`, `\leq`, `\geq`, `\neq`,
      `\equiv`), a binary operator (`+`, `-`, `\times`, `\cdot`, `\pm`),
      or a term that obviously continues the first expression, the two are one
      equation — remove the delimiter pair between them and join the content
      with a line break. Two independent back-to-back equations stay separate.

    Convert all of them, not the first few, and do this even when the page has
    other things wrong with it.

    REQUIRED — MATH THAT ARRIVED WITH NO DELIMITERS AT ALL

    Some pages come through with their mathematics as plain text: "969. x⁴ when
    x = 3", "1011. -200 + 65". Nothing above catches these, because there are
    no delimiters to convert. Wrap each such expression in `$ … $`.

    THE TEST. A span is mathematics when it has EITHER of:
    - a symbol standing for a quantity: `9x + 7`, `-m`, `12n`, `x⁴`;
    - an operator or relation joining quantities: `25 - 7`, `-200 + 65`,
      `20 ÷ 4`, `42 ≥ 27`, `y - 8 = 32`;
    or when it is written in notation that is only ever mathematical, such as
    absolute-value bars: `|7|`, `|-25|`, `|8 - 4|`.

    A bare quantity with neither a symbol nor an operator is NOT mathematics
    here. Leave these exactly as they are:
    - an item's own number: `925.`, `1005.` — never wrap it, and never let it
      join the expression after it;
    - a part marker: `ⓐ`, `(a)`, `a)`;
    - a lone value or a row of them, which in drill exercises is the exercise's
      subject matter, not an expression: `407 8,564`, `864,951`, `1430`;
    - page numbers, years, section and theorem numbers, and any other
      identifier the document refers to elsewhere by name.

    Wrap each maximal expression on its own, not the sentence around it:
    "evaluate 9x + 7 when x = 3" becomes "evaluate $9x + 7$ when $x = 3$" —
    two spans, with the prose between them untouched.

    WHEN YOU ARE NOT SURE, LEAVE IT BARE. The two mistakes are not equal: a
    span wrongly left alone is the page as it already stands and any later pass
    can still find it, while a number wrongly wrapped is an identifier
    corrupted — and an exercise stripped of the number the rest of the book
    cites it by cannot be recovered downstream.

    REQUIRED — A DOLLAR SIGN THAT IS NOT A DELIMITER

    `$` also means money, and word problems are full of it: "The skirt cost
    $15 more than the blouse." Escape every such dollar sign as `\$`.

    This matters most on the lines you have just edited. A page that mentions
    a price and says nothing in mathematics survives its stray `$`, but as
    soon as this pass writes real delimiters nearby, a reader counting from
    the left pairs the money sign with one of them and takes the prose
    between for an expression — "$15 more than the blouse. Let $" becomes
    mathematics. Escape the currency whenever a line carries both, and escape
    it on sight even when it does not: the delimiters that collide with it may
    be written later, on a page you no longer have in front of you.

    A dollar sign that opens or closes real mathematics is never escaped.

    REQUIRED — NO MATHEMATICAL NOTATION IN UNICODE, ANYWHERE

    Every piece of mathematical notation on the page is written in LaTeX. NOT
    ONE Unicode mathematical character survives this pass — not in prose, not
    in a heading, not in a list item, not in a table cell, not in a caption:

    - superscripts: `x⁴` -> `x^4`, `3ˣ` -> `3^x`, `x¹⁰` -> `x^{10}`
    - subscripts: `R₁` -> `R_1`
    - operators and relations: `×` -> `\times`, `÷` -> `\div`, `·` -> `\cdot`,
      `±` -> `\pm`, `≤` -> `\leq`, `≥` -> `\geq`, `≠` -> `\neq`, `√` -> `\sqrt`,
      `∞` -> `\infty`, `→` -> `\to`, `⇒` -> `\Rightarrow`,
      `⇔` -> `\Leftrightarrow`, `∈` -> `\in`, `⊆` -> `\subseteq`,
      `∪` -> `\cup`, `∩` -> `\cap`, `∀` -> `\forall`, `∃` -> `\exists`,
      `∅` -> `\emptyset`, `∑` -> `\sum`, `∏` -> `\prod`, `∫` -> `\int`,
      `∂` -> `\partial`, `∇` -> `\nabla`, `≈` -> `\approx`,
      `≡` -> `\equiv`, `∼` -> `\sim`, `∘` -> `\circ`, `⊥` -> `\perp`
    - Greek letters used as symbols: `α` -> `\alpha`, `π` -> `\pi`,
      `Ω` -> `\Omega`
    - anything else of the same kind: if a character is notation rather than
      a word, it has a LaTeX spelling and that spelling is what you write.

    A GLYPH IS ITSELF THE EVIDENCE. Finding one of these outside a math span
    does not mean leaving it alone — it means you have found mathematics that
    was never delimited. Convert the notation AND wrap it, under the wrapping
    rule above: `x⁴` in the middle of a sentence becomes `$x^4$`, and
    `α-mixing` becomes `$\alpha$-mixing`.

    This changes how the notation is ENCODED, never what it says: `x⁴` and
    `x^4` are the same power. Do not go further and rewrite the mathematics
    itself — do not simplify, reorder, factor, evaluate, or "tidy" an
    expression into a form you prefer.

    WHAT IS NOT NOTATION, and is therefore left exactly as written:
    - A letter inside a word or a name. `Pólya`, `Erdős`, `café`, `Ω` when it
      is a person's initial. Respelling a proper noun changes a fact.
    - A reference marker. A superscript that points at a footnote is a
      pointer, not an exponent: `Theorem 2¹` keeps its `¹`.
    - Ordinary punctuation and typography — dashes, curly quotes, ellipses,
      non-breaking spaces. These are not mathematics and get no LaTeX.
    - Anything inside code or verbatim content, where every character is
      already literal.

    Otherwise change the delimiters only, never the expression between them.

    ALSO STANDARDISE

    - Headings. Mark every heading with `#`s, one level per structural level,
      deepening consistently down the page. A heading written as a line of text
      underlined by `===` or `---` on the next line is a heading: replace both
      lines with a single `#`-marked one. Do not invent a heading, remove one,
      or promote a line that is not one.
    - Lists. `-` for bullets and `1.` numbering for ordered lists, with nesting
      shown by indentation. Keep every item, its position, and any label the
      document gives it.
    - Part markers. Textbooks letter an exercise's parts in whatever glyph the
      typesetter had — `ⓐ`, `(a)`, `a)`, `a.` — and one page often mixes
      several. Write them all one way: `(a)`, `(b)`, `(c)`.
      Standardise the DECORATION only. The letter itself is the part's
      identity, referred to elsewhere as "by part (b)", so `ⓑ` becomes `(b)`
      and never `(a)`, never a bullet, and never nothing. A part marker is not
      mathematics: it takes no `$` and no LaTeX.
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
    - Mathematical content. What may change is the delimiters around an
      expression, and the ENCODING of its notation where that notation is a
      Unicode look-alike for LaTeX (see the Unicode rule above). Nothing else:
      never rewrite the expression, never convert notation to a form you
      prefer, and never alter a symbol the document chose.
    - Notation, terminology, and spelling. Keep the document's own conventions
      and symbols; standardise the markup, not the author.
    - Numbering and labels. Leave every identifier the document uses — section
      and theorem numbers, exercise numbers, part letters — exactly as written.
      They are referred to elsewhere by name. Standardising a part marker's
      decoration (see above) is the one permitted change and does not touch
      the identifier: `ⓑ` and `(b)` are both part b. Never renumber, never
      re-letter, never drop a label.
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
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.formatter = dspy.Predict(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, markdown: str) -> str:
        """Format one page.

        Args:
            markdown: The page's corrected markdown.

        Returns:
            The page's markdown with standardised formatting.
        """
        result = await self.formatter.acall(markdown=markdown)
        if self._recorder:
            self._recorder.record('formatter', {'markdown': markdown}, result)
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
        module: The formatting module.
    """

    def __init__(self, module: Formatter) -> None:
        self.module = module

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
