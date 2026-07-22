# Book fixtures (PDF slices)

Small page-slices of real math textbooks, committed so splitter/distributor
stress tests reuse them without re-downloading the full (50–100 MB) books.

Each slice was cut from the openly-licensed source PDF with `pypdfium2`
(`PdfDocument.import_pages`); pages are **0-based** as in the source file.

| Fixture | Source | Pages (0-based) | Why it's here |
|---|---|---|---|
| `ea2e_ch1_review.pdf` | OpenStax *Elementary Algebra 2e* | 186–188 | Chapter-1 **Review Exercises**: many short "In the following exercises, …" lead-ins, each governing only 2–6 numbered exercises before the next lead-in, with sub-section headers ("Identify Multiples and Factors", …) interleaved. Hardest case for the **distributor** — over-extension across the next lead-in/header is the failure mode. |
| `ea2e_sec1_3_exercises.pdf` | OpenStax *Elementary Algebra 2e* | 69–70 | A full **section exercise set** (§1.3): a `SECTION 1.3 EXERCISES` / "Practice Makes Perfect" header, dense lead-in runs, then an **"Everyday Math"** block whose word problems (255–257) carry their **own embedded directive** ("Use integers to write …"). Distributor must NOT treat those embedded directives as run-governing lead-ins. |

## Source & licence

OpenStax *Elementary Algebra 2e* — © Rice University, CC BY 4.0.
Book page: https://openstax.org/details/books/elementary-algebra-2e
Full PDF: `https://assets.openstax.org/oscms-prodcms/media/documents/elementary-algebra-2e_-_WEB.pdf`

## Re-slicing / adding fixtures

```python
import pypdfium2 as p
src = p.PdfDocument("elementary-algebra-2e_-_WEB.pdf")   # full download, not committed
out = p.PdfDocument.new()
out.import_pages(src, [186, 187, 188])                    # 0-based page indices
out.save("tests/fixtures/books/ea2e_ch1_review.pdf")
```

To find lead-in-dense pages in a fresh book, scan page text for
`in the following exercises` (case-insensitive) and rank by count.
