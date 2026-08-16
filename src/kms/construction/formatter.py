"""Formatting pass: markdown conventions, math delimiters, lists."""

import logging

import dspy
from langgraph.types import Send

from kms.core import models, module, state
from kms.core.edits import LineEdit, apply_line_edits, number_lines

logger = logging.getLogger(__name__)


class Signature(dspy.Signature):
    r"""
    You are a meticulous formatter of document transcriptions. You are given
    one page of a document as numbered lines of markdown. For each line that
    needs a formatting change, emit an edit describing exactly that change. A
    line you do not mention is left exactly as it is.

    Change how the content is written down, never what it says. Every word,
    number, symbol, and mathematical expression must survive unchanged — only
    the markup around them may change. Where a change would alter meaning, or
    where you cannot make it without guessing at meaning, leave the line as
    it is.

    EDIT FORMAT

    Each line is shown prefixed with its 1-based line number in square
    brackets, as `[14] some text`. For every line you change, emit one edit
    with two fields:

    - index: the line's number.
    - replacement: the full new text of that line. An empty replacement
      deletes the line. A replacement that contains line breaks inserts
      several lines in its place.

    To wrap a multi-line display in `$$ … $$`, edit the first line with a
    replacement that begins with `$$` and the last line with a replacement
    that ends with `$$`. To delete a heading's underline, replace the
    underline line with an empty replacement and rewrite the heading line
    with `#`s.

    Emit an edit ONLY for a line that changes. Never emit an edit for a line
    that already follows the rules. If no line changes, return an empty list.
    Return edits in line order, each index used at most once.

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

    Return only the list of edits. If the page already follows the conventions
    above, return an empty list.
    """

    lines: str = dspy.InputField(
        description='The page as numbered lines of markdown, one line per '
        'row, each prefixed with its 1-based line number in square brackets.'
    )
    edits: list[LineEdit] = dspy.OutputField(
        description='One edit per changed line, in line order: the line '
        'number and the replacement text.'
    )


class Formatter(module.Module):
    """Applies the document's markdown conventions to one page."""

    signature = Signature
    record_name = 'formatter'

    def encode(self, markdown: str) -> dict:
        """Builds the formatter-signature kwargs for one page."""
        return {'lines': number_lines(markdown)}

    def decode(self, prediction, **inputs) -> str:
        """Returns the page with its formatting edits applied."""
        return apply_line_edits(
            inputs['markdown'], module.as_list(prediction.edits)
        )


class FormatterNode:
    """Langgraph node dispatching one formatter worker per segment."""

    def __init__(self, module: Formatter) -> None:
        self.module = module

    def dispatch(self, state: state.State) -> list[Send] | str:
        """Sends one worker per segment with content, else the collector."""
        segments = state.get('segments', [])
        sends = [
            Send('formatter_worker', {'segment': segment})
            for segment in segments
            if segment.content
        ]
        return sends or 'formatter_collect'

    async def worker(self, state: dict) -> dict:
        """Formats one segment's content."""
        segment: models.Segment = state['segment']
        formatted = await self.module.aforward(markdown=segment.content)
        return {'format_results': [(segment.index, formatted)]}

    def collect(self, state: state.State) -> dict:
        """Merges per-page format results back onto the segments."""
        results = state.get('format_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'content'
        )
        logger.info('formatter: %d page(s) formatted', len(results))
        return {'segments': segments}
