# Pipeline extraction robustness test

**Date:** 2026-07-23 · **Scope:** analysis only, no pipeline fixes (per request).
**Goal:** run a variety of book styles / levels / math fields front-to-back and report on
how robustly the extraction pipeline holds up. All inputs and outputs are kept in a
DSPy-trainable shape (see [Training data](#training-data-captured)).

## Method

Six openly-licensed textbooks were sliced to small page ranges chosen to exercise the
whole pipeline (exposition with definitions/theorems/proofs **and** exercise blocks,
plus figures/tables where the book has them), deliberately covering fields and styles
**not** in the existing `tests/fixtures/books/` corpus (which already covers elementary
algebra, calculus, real analysis, metric spaces, linear algebra).

Each book was run end-to-end (`kms.cli`) with `KMS_TRACE_DIR` set so every DSPy stage's
per-call I/O was captured. Runs were **DB-less** (Neo4j `node_persister` is a documented
no-op when `NEO4J_*` is unset) — Aura's Bolt port is blocked from this sandbox, and graph
persistence is orthogonal to extraction.

## Corpus & extraction summary

| Book | Field | Pages | Nodes | Def | Thm | Prob | Figs | Tbl rows | `$$` | Instr stamped |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Stein, *Elem. Number Theory* | Number theory | 45–48 | 47 | 1 | 4 | 0 | 0 | 0 | 14 | 0 |
| Hammack, *Book of Proof* | Logic / proof | 58–61 | 50 | 0 | 0 | 16 | 0 | 28 | 4 | 14 |
| Grinstead & Snell, *Intro. to Probability* | Probability | 24–28 | 67 | 7 | 0 | 5 | **0** | 0 | 36 | 0 |
| Lebl, *diffyqs* | ODEs | 28–31 | 51 | 0 | 1 | 21 | 4 | 0 | 14 | 0 |
| Morris, *Topology Without Tears* | Topology | 41–44 | 37 | 1 | 0 | 8 | 0 | 0 | 16 | 0 |
| Levin, *Discrete Math* (graph theory) | Combinatorics | 125–128 | 50 | 8 | 2 | 3 | 5 | 8 | 6 | 0 |

Provenance / integrity across **all six** books: **0** dangling member refs, **0**
duplicate-member problem groups, **0** empty-content nodes, **100%** field coverage. The
granularity bug the splitter was built to fix does not reappear on any book.

## What proved robust (across all styles)

1. **Structural extraction + provenance are rock-solid.** Every entity's `members` point
   at real nodes; no duplicate-membered exercise groups anywhere (the pre-splitter failure
   mode). The flat node stream + sparse overlay model held on every book.
2. **Entity typing generalizes.** `theorem` correctly subsumed Proposition / Lemma /
   Corollary (Stein §2.5: Prop 2.5.3/2.5.5, Lemma 2.5.7, Thm 2.5.8; Levin: "Handshake
   Lemma"). Worked examples were captured as `problem` (Lebl, topology) per the AutoMathKG
   model.
3. **Inline / unnumbered definitions are caught.** In Levin's graph-theory section, 6 of 8
   definitions are inline bold terms ("a *simple graph*…", multigraph, bipartite, degree)
   with no "Definition N.N" label — all found, with `number=None` correctly left unset.
4. **LaTeX / math fidelity is high.** Dense notation survives the OCR→correction→extract
   path, e.g. Stein's `$(\mathbf{Z}/p\mathbf{Z})^*$`, 14–36 display-math blocks and
   hundreds of inline spans per slice, delimiters normalized to `$`/`$$`.
5. **Table OCR is excellent.** Hammack's **6-column truth tables** render as clean markdown
   with `$T$`/`$F$` cells and bolded result columns (`$\mathbf{F}$`) — 28 table rows across
   the slice, structurally intact.
6. **The splitter atomizes packed exercise runs.** Lebl's big exercise blocks became 21
   distinct `problem` entities; Hammack's became 16 — each atomic, precise provenance, no
   duplicates.
7. **The instruction distributor is correct on both regimes.** On the one lead-in-heavy
   book (Hammack) it segmented **two distinct lead-in groups** — "Without changing their
   meanings, convert each of the following sentences…" over the word-problems, then "Write a
   truth table…" over the next group (14/16 stamped). On the self-contained styles (Stein,
   Lebl, topology) it correctly stamped **nothing** — no false governance.
8. **Attribute coverage is high.** number/title/field/contents near-complete; proofs
   captured (Stein 4/4); solutions where a worked example shows one.

## Robustness gaps found

These are reported, **not fixed** (per request).

### 1. Figure extraction is book-dependent — probability figures dropped entirely
The Grinstead & Snell slice references figures in-text — "…which direction to walk next
(see Figure 1.6a)…", "…refer to Figure 1.7 for Venn diagrams…" — but the run produced
**zero image nodes** and zero `![N]()` markers. Contrast Lebl ODE (4 figures: direction
fields) and Levin graph theory (5 figures: graph diagrams), where figures were captured
fine. The likely cause is front-end: G&S is an old dvips/PostScript book whose figures are
vector line-art, which Mistral OCR does not appear to return as cropped raster figures,
whereas the newer PDFs' figures come through. This is the one place content was silently
lost. Worth a targeted front-end look (Mistral OCR options for vector figures, or a
fallback rasterizer) — it is upstream of the entity layer, which behaved correctly given
what it received.

### 2. The `FIELDS` taxonomy is too coarse for a math KG
`FIELDS` has 7 buckets (algebra, geometry, analysis, logic, probability and statistics,
applied mathematics, foundations of mathematics). Several whole fields have **no home**, so
content is forced into the nearest bucket, lossily and inconsistently:
- Number theory (Stein) → **algebra** (all 5).
- Topology (Morris) → **foundations of mathematics** (all 9).
- Graph theory (Levin) → **scattered** across applied mathematics (8), foundations (2),
  algebra (2), logic (1) — the same section split four ways.
- ODEs (Lebl) → **analysis** (21) vs applied mathematics (1) — defensible but arguably
  inverted for differential equations.

Field labels are always *populated* (100%) and never crash, but they are not reliable as a
KG facet for number theory / topology / combinatorics. This is a taxonomy-design decision
to revisit, not an extraction bug.

### 3. Field can key off embedded content rather than the item's nature (minor)
Hammack's logic exercises use math sentences as examples ("A matrix is invertible iff…"),
and 3/16 were fielded `algebra`/`analysis` instead of `logic` — the classifier keyed off
the embedded statement's topic. Ambiguous rather than wrong.

## Per-book notes

- **Stein (number theory).** Clean §2.5 exposition: 1 definition + 4 theorem-family
  entities, all with proofs. The only "Exercise" strings are in-text cross-references ("see
  Exercise 2.28"); the finder correctly produced **0 problems** and did **not** mistake the
  cross-reference numbers for entity numbers (the attributor's known number-format hazard did
  not trigger here).
- **Hammack (logic).** Best table result: multi-column truth tables intact; distributor
  segmented two lead-in groups correctly. No defs/theorems (correct — pure exercise section).
- **Grinstead & Snell (probability).** Text + definitions + worked examples extracted well
  (7 defs, most unnumbered → correct); **figures lost** (gap #1).
- **Lebl (ODEs).** Heaviest slice: 21 atomic exercises from packed blocks + 4 direction-field
  figures captured. Self-contained exercises → distributor correctly silent.
- **Morris (topology).** §1.3: 1 definition + worked examples + Exercises 1.3 (items 1–6).
  0 theorems is **correct** — the only "Proposition 1.3.6" on the page is an in-text
  reference, not a stated theorem.
- **Levin (graph theory).** Richest entity mix: 8 definitions (mostly inline), 2 theorems
  (Handshake Lemma + odd-degree parity), 3 problems, 5 graph figures, a table.
  *(Corpus note: the first pick for this slot, pages 49–53, turned out to be Levin's
  propositional-logic chapter — re-sliced to the graph-theory chapter for genuine
  combinatorics coverage.)*

## Training data captured

Every DSPy signature in the pipeline was instrumented with `tracing.record()` for this run
(previously only splitter / instruction_finder / distributor were). Captured stages:
`corrector, extractor, seam_merger, splitter, instruction_finder,
{definition,theorem,problem}_finder, definition_{identify,bodylist},
theorem_{identify,statement_bodylist,proof_bodylist}, problem_identify, distributor`.

**227 trace lines** were captured (per-call `{stage, inputs, outputs}` JSONL). These feed
the existing loaders in `training/*/dataset.py`, which wrap them as `dspy.Example` at load
time (the on-disk shape is JSONL by design; DSPy `Example`s are built in memory) — e.g.
`training/splitter/dataset.py::load_windows("robustness_test/traces/<book>/splitter.jsonl")`
and `training/distributor/dataset.py::load_runs(["robustness_test/runs/<book>", …])`.

### Layout

```
robustness_test/
  books/            6 input PDF slices (kept)
  runs/<book>/      document.md + entities.json + nodes.json (kept); Segments/ page
                    renders are gitignored (intermediate, ~9.5M, reconstructable from books/)
  runs/<book>.log   full stdout/stderr of each run
  traces/<book>/    <stage>.jsonl per DSPy signature (the training data)
  REPORT.md         this file
```

Per-book / per-stage trace counts:

| Book | corr | extr | seam | split | instr.find | def.find | thm.find | prob.find | def.id | def.body | thm.id | thm.stmt | thm.proof | prob.id | distrib | Σ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nt_stein_congruences | 4 | 4 | 3 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 4 | 4 | 4 | 0 | 0 | 35 |
| logic_hammack_truthtables | 4 | 4 | 3 | 2 | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 16 | 2 | 39 |
| prob_grinstead_snell | 5 | 5 | 4 | 2 | 2 | 2 | 2 | 2 | 7 | 7 | 0 | 0 | 0 | 5 | 0 | 43 |
| ode_lebl_diffyqs | 4 | 4 | 3 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 21 | 0 | 39 |
| topology_morris | 4 | 4 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 8 | 0 | 26 |
| combinatorics_levin | 4 | 4 | 3 | 2 | 2 | 2 | 2 | 2 | 8 | 8 | 2 | 2 | 1 | 3 | 0 | 45 |

**Note on `corrector` traces:** the page image input is recorded as a placeholder, not the
bytes — the image is large and reconstructable from the input PDF; the trainable text signal
(transcription → corrected) is captured in full.

## Sources & licences (all permit excerpt redistribution)

- Stein, *Elementary Number Theory* — see wstein.org/ent.
- Hammack, *Book of Proof* — CC-licensed (richardhammack.github.io/BookOfProof).
- Grinstead & Snell, *Introduction to Probability* — GFDL (dartmouth `~prob`).
- Lebl, *Notes on Diffy Qs* — CC BY-SA / CC BY-NC-SA (jirka.org/diffyqs).
- Morris, *Topology Without Tears* — freely distributed (topologywithouttears.net).
- Levin, *Discrete Mathematics: An Open Introduction* (4e) — CC BY-SA
  (discrete.openmathbooks.org).

## Caveats

- Small slices (4–5 pages) per book — a breadth-over-depth robustness sweep, not an
  exhaustive per-book audit.
- Runs are DB-less; the graph tier's structural provenance layer was not exercised here.
- "Correct" judgements above are from reading `document.md` + `entities.json` + `nodes.json`
  against the source pages, not against a hand-labeled gold set.
