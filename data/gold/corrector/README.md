# Corrector gold set

Hand-verified `(page image, OCR transcription) -> corrected transcription`
triples for the correction pass (`kms.ingestion.corrector`), built to be the
train/dev material for a later MIPROv2 run. **This directory is data only** —
there is no optimizer here, just the examples, their provenance, and an
annotation of every difference between input and gold.

69 records over 38 pages from the 12 committed book fixtures:

| | records | pages with a correction | corrections |
|---|---|---|---|
| `real` — genuine Mistral output, gold = the hand-verified page | 38 | 7 | 11 |
| `perturbed` — realistic errors injected into a verified page | 31 | 31 | 64 |

The two halves teach opposite halves of the job. Mistral is *good*: 31 of 38
pages came back with nothing that changes meaning, so their gold is the
transcription verbatim. Those records are what keeps an optimized prompt from
inventing corrections. The perturbed records supply the density of genuine
errors that 38 real pages do not.

## Provenance

- **Transcriptions**: real Mistral `mistral-ocr-latest` output, harvested
  2026-07-30 through `kms.ingestion.ocr` with the pipeline's own settings
  (`extract_header` / `extract_footer` on, figure refs rewritten to the
  positional `![N]()` convention). `real/*/transcription.md` is the page
  markdown exactly as the API returned it — untouched, including its
  `\(…\)`/`$…$` delimiter mix and its whitespace.
- **Pages**: `tests/fixtures/books/*.pdf`, openly licensed textbook slices
  (see that directory's README for sources and licences). Elementary algebra
  through graduate topology, plus truth tables, probability tables, sage
  sessions, and multi-column exercise grids.
- **Page images**: *not committed* — 13 MB of PNGs that are reproducible from
  the fixtures. `render_pages.py` rasterizes them at the same scale the OCR
  front-end uses (`ocr.RENDER_SCALE = 2.5`).
- **Gold**: written by hand (Claude, in a Claude Code session), reading each
  page image against its transcription under the rules in the corrector's own
  prompt. No teacher model was involved — an earlier deleted
  `kms.training.generate_corrector_data` had a strong LM produce the gold; this
  set deliberately does not, so it can be used to *judge* models rather than
  inherit one's mistakes.

## Layout

```
index.json                     # the manifest — everything below is described here
real/<book>_pNN/               # one directory per source page
  transcription.md             #   Mistral's output, verbatim
  corrected.md                 #   the gold
perturbed/<book>_pNN_v1/       # errors injected into the verified page
  transcription.md             #   gold + injected errors
  corrected.md                 #   the gold, byte-identical to the real record's
render_pages.py                # page images <- fixture PDFs
```

Each `index.json` record:

```json
{
  "id": "nt_stein_congruences_p00",
  "kind": "real",
  "book": "nt_stein_congruences",
  "source_pdf": "tests/fixtures/books/nt_stein_congruences.pdf",
  "page": 0,
  "render_scale": 2.5,
  "page_image": "nt_stein_congruences_p00.png",
  "split": "train",
  "transcription": "real/nt_stein_congruences_p00/transcription.md",
  "corrected": "real/nt_stein_congruences_p00/corrected.md",
  "edits": [{"class": "quantity", "before": "...", "after": "...", "note": "..."}],
  "edit_classes": ["position", "quantity"]
}
```

`edits` always reads transcription → corrected, for both kinds, so a metric or
an error analysis can ask "which classes does this prompt actually catch?"
without diffing anything. Perturbed records also carry `derived_from`.

**Splits**: `train` 46 / `dev` 23, assigned by *page* — a perturbed record is
always in the same split as the page it was derived from, because its gold text
is that page's gold text, and splitting them would leak a dev answer into a
train demo. Every one of the 12 books appears in both splits.

## Loading

```sh
uv sync --extra mistral
python data/gold/corrector/render_pages.py        # -> output/gold_pages/
```

```python
import json
from pathlib import Path

import dspy

GOLD = Path('data/gold/corrector')
PAGES = Path('output/gold_pages')

index = json.loads((GOLD / 'index.json').read_text())
examples = {'train': [], 'dev': []}
for record in index['records']:
    examples[record['split']].append(
        dspy.Example(
            page_image=dspy.Image(str(PAGES / record['page_image'])),
            transcription=(GOLD / record['transcription']).read_text(),
            corrected=(GOLD / record['corrected']).read_text(),
        ).with_inputs('page_image', 'transcription')
    )
```

The field names are the corrector `Signature`'s, so the examples drop straight
into `Corrector` — and into `kms.core.recording`'s shape, if it is ever
convenient to merge recorded pipeline runs with this set.

## What "corrected" means here

The gold applies the corrector prompt as written: correct only what the image
settles and only where the meaning changes; leave everything else, including
the document's own errors. In practice that meant a few boundaries had to be
decided, and they were decided the same way on every page:

- **The source's own errors stay.** "not one-to one" (Morris), "The polynomials
  x² − 1 has four roots" (Stein), a lead-in with a doubled comma (OpenStax, and
  Mistral silently *fixed* that one — the gold keeps Mistral's version, since
  punctuation that changes no meaning is out of scope in both directions).
- **Markdown structure is frozen**, even where it overreaches: Mistral typed a
  bold "Proof." run as an H1 and a two-column definition list as a table. The
  prompt puts markdown structure under Formatting, so the gold leaves both.
  Indentation *inside code* is the documented exception and was corrected.
- **The page's footer is part of the transcription.** `ocr.py` appends each
  page's extracted footer back onto its markdown, so whatever sits below the
  body — a footnote, a colophon, a licence line — arrives as a trailing block.
  Eleven of the 38 pages carry one: 4 real footnotes (Levin ×3, Morris ×1) and 7
  pieces of page chrome. **Gold keeps all of them.** A footnote is content, and
  chrome is the *extractor's* to discard (it types such blocks `furniture`), not
  the corrector's — this pass never deletes.

  > **How these 11 were patched.** The set was first harvested when
  > `build_segments` dropped the footer field, so those transcriptions were
  > missing it. Rather than re-OCR — which returns different markdown run to run
  > and would have invalidated the 27 unaffected pages' verified gold — the
  > footer was appended from the *same* saved OCR response that produced the
  > committed markdown, i.e. that response read through today's code path. The
  > appended text was then re-verified against each page render like any other
  > content, which is where the Book of Proof licence correction came from.
- **Reading order across columns is a correction.** Two OpenStax pages were read
  column-major out of a numbered multi-column grid (`1005, 1008, 1006, 1007`).
  The page numbers its items across each row, so the transcribed sequence is not
  the page's own order; the gold resequences the blocks and changes no exercise
  text. This is the one class where the gold reorganises anything.

## Coverage

Corrections by class, over both halves (the classes are the prompt's own):

| class | corrections | class | corrections |
|---|---|---|---|
| quantity | 19 | presence | 8 |
| substitution | 14 | position | 5 |
| polarity | 10 | attachment | 4 |
| extent | 6 | order | 3 |
| relation | 6 | | |

The eleven genuine (non-injected) corrections are worth reading on their own —
they are what this OCR actually gets wrong:

| record | class | what happened |
|---|---|---|
| `nt_stein_congruences_p00` | quantity | a digit dropped from the 27-digit 2⁸⁹−1 in a sage transcript |
| `nt_stein_congruences_p00` | position ×2 | code indentation flattened, moving a `print` out of its `if` and an assignment out of its `for` |
| `nt_stein_congruences_p02` | presence ×2 | `R.<x>` read as an HTML tag, emitting two stray `</x>` |
| `logic_hammack_truthtables_p00` | relation | "the truth table for ⇒" transcribed as "⇔", making the sentence cite the table it derives |
| `lebl2_metricspaces_sec8_1_exercises_p00` | extent | slanted fraction flattened: `(1/(n+1), 1/n)` became `(1/n+1, 1/n)` |
| `logic_hammack_truthtables_p00` | substitution | licence badge reads BY-NC-ND; transcribed as `CC BY-NC-SA` |
| `logic_hammack_truthtables_p02` | substitution | the same badge, the same misreading |
| `ea2e_ch1_review_p02` | order | three-column exercise grid read column-major |
| `ea2e_sec1_3_exercises_p01` | order | two-column word-problem block read column-major |

Note what is *absent*: across 38 pages of real mathematics there was not one
misread subscript, exponent, or root index — the failure modes the corrector's
docstring is built around. Mistral's residual errors here are structural
(indentation, column order, markup artifacts), long-numeral, and — once the page
footer became part of the transcription — the licence badge it reads off an
image rather than off text. The perturbed
half covers the notational classes anyway, so an optimized prompt is scored on
both, but the real distribution is a finding in itself.

## Perturbed records

Each perturbed record takes one verified page and injects one to three errors
that are *visible against that page's image* — a real digit changed, a negation
dropped, two table cells swapped between columns, a term removed from a sum, two
rows of an eight-row truth table transposed, a page's whole footnote deleted, a
year and an author's name altered inside a citation. Every injection contradicts either
the image or the page's own surrounding text, so a correct model has evidence to
find it. Injections are exact string substitutions applied to the gold, each
asserted to match exactly once (a deletion is written as an anchored swap, so
the annotation also records where the missing text belongs), so the pair differs *only* by the annotated
edits and every perturbed gold is byte-identical to its real counterpart's.

They are labelled `kind: "perturbed"` precisely so they can be down-weighted,
held out, or dropped: they are synthetic, their error density (100% of records)
is far above the real rate (18% of pages), and an optimizer trained only on them
would learn that something is always wrong.
