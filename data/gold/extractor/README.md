# Extractor discard gold set

What the extractor must **throw away** and what it must **keep**, over 22 real
pages. **This directory is data only** — no runner, no metric, no code. Nothing
here executes; it is a labelled answer key for the one decision in the pipeline
that cannot be undone.

## Why this exists

`kms.ingestion.extractor` types page apparatus as `furniture` and drops those
blocks inside the stage — no node, no id, no graph vertex, nothing downstream
that could notice. A wrong drop deletes content silently, and there is no other
record of it than a DEBUG log line.

The judgement is the LLM's and should stay the LLM's: "is this text about the
book as an object?" is language, and a rule written against one publisher's
colophon fails on the next. What this set does is hold that judgement to
account, so a prompt edit can be checked instead of eyeballed.

It exists because eyeballing already failed once. A first attempt at the
page-opening-fragment guard produced **zero furniture drops** — correct by the
measure being watched — while the model quietly omitted the fragment instead.
Deleted and never-emitted look identical from outside the stage. Only a page
with a known answer separates them.

## Shape

`index.json` holds one record per page:

```json
{
  "id": "hefferon_p0476",
  "book": "hefferon",
  "source": "Jim Hefferon, *Linear Algebra*, 4th edition (CC BY-SA)",
  "page": 476,
  "markdown": "pages/hefferon_p0476.md",
  "discard": [],
  "keep": ["39368 -1", ">norm(v10)/norm(v9)"],
  "note": "THE REGRESSION CASE. The page opens mid-listing …"
}
```

- **`markdown`** — the page exactly as the extractor receives it: Mistral OCR
  output with the page's footer appended and figure refs rewritten to `![N]()`,
  i.e. `ocr.build_segments` output. Self-contained; unlike the corrector set
  this needs no page images, because the extractor never sees one.
- **`discard`** — verbatim spans that must NOT survive into the node stream.
- **`keep`** — verbatim spans that MUST survive. These are the traps: content
  that resembles apparatus, and apparatus-adjacent content that has to come
  through untouched.

Gold is **spans, not block indices**, deliberately. Block segmentation varies
between runs on the same page (an Octave session came back as one `code` node
once and four the next time), so an index-based answer key would fail on
formatting noise rather than on the judgement being measured. Substring
containment over the kept nodes' text is stable against that.

Every span is asserted to be a substring of the committed page when the set is
built, so a typo cannot silently weaken a case into a no-op.

## Coverage

22 pages: **12 with something to discard, 10 controls** with nothing at all.
33 discard spans, 27 keep spans.

| kind of apparatus | pages |
|---|---|
| running foot / colophon repeated on every page | 6 |
| title page | 3 |
| copyright page — licence terms, funding, typesetting, trademarks | 2 |
| publisher advertisement | 1 |
| controls: body prose, exercises, contents, figures with captions | 10 |

The controls carry the weight here. A discard rule is easy to satisfy by
dropping more, so most of this set is pages where the right answer is *nothing*.

### The two cases that motivated it

- **`hefferon_p0476`** — the page opens mid-listing with the tail of an Octave
  session, `39368 -1`. A bare numeric line at the top of a page is
  indistinguishable from a folio and was being deleted, *before* the seam merger
  — the one stage that could have rejoined it to its other half.
- **`openstax_ea2e_p0007`** — the bare chapter numerals beside each contents
  heading look exactly like folios. They are chapter numbers and belong to the
  contents; only the colophon goes.

A third pins a boundary: **`openstax_ea2e_p0958`** contains
`![Right arrow icon]()`, image markup with no extracted figure behind it, inline
in body prose. It is a `keep` — body text is never apparatus, so the discard
rule must not reach into the running text even when what it finds there is junk.

## Boundary calls

Made once and applied to every page, so the set is consistent even where the
question is genuinely arguable:

- **Title and copyright pages are apparatus.** Title, edition, author credits,
  publisher address, ISBN, printer's key, licence terms, typesetting note,
  funding acknowledgement, trademark notice. All of it describes the book as an
  object, which is the test the prompt states. Note the consequence: book
  metadata does not survive ingestion, so anything wanted on `:Source` has to be
  captured before this stage.
- **A standalone figure placeholder is never apparatus.** Front-matter pages
  carry a real extracted crop (cover art, publisher logo) as its own block;
  those are `keep`. This does not contradict the prompt's rule that a
  placeholder *inside* a line of apparatus text belongs to that furniture block
  — the distinction is standalone block versus mixed into a text run, which is
  observable on the page.
- **An advertisement is apparatus** even though it is not a running foot.
- **Borderline, and recorded as such:** a cover-art note that names a painter
  and a geometric construction. It teaches something, but it is about the cover,
  so it goes.

## Provenance

Pages were sampled from four openly licensed books, weighted toward front matter
where apparatus is densest, and OCR'd on 2026-07-30 with `mistral-ocr-latest`
under the pipeline's own settings. Sources are named per record:

- **OpenStax**, *Elementary Algebra 2e* — CC BY 4.0
- **Jim Hefferon**, *Linear Algebra*, 4th edition — CC BY-SA
- **Jiří Lebl**, *Basic Analysis I* — dual CC BY-NC-SA / CC BY-SA

> **One book was sampled and then excluded.** Hammack's *Book of Proof* has the
> most instructive apparatus in the corpus — a running foot in four different
> OCR spellings, including the licence badge whose stray image placeholder was
> the leak that prompted the `furniture` type. It is CC BY-NC-**ND**, and
> `tests/fixtures/books/README.md` excludes NoDerivs works by policy ("Treil …
> was **excluded** — it is CC BY-NC-**ND**, whose NoDerivs term makes
> redistributing excerpts unsafe"), so its pages are not committed here.
> The badge-inside-apparatus case is therefore **not covered by this set**.

The judgement was made from the page text rather than a page render, which is
correct for this decision: the question is what the text is *about*, not whether
it matches the page. That is the corrector's question, and it has its own set.

## Using it

There is no runner. To check a prompt change by hand: run the extractor over a
record's `markdown`, concatenate the content of the nodes it keeps, and confirm
every `discard` span is absent and every `keep` span present. A false drop — a
`keep` span missing — is the serious failure and should be weighted accordingly;
a missed discard leaves a tidy line in the document and nothing worse.
