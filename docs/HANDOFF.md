# KMS — Handoff

Knowledge-management pipeline that turns a math textbook PDF into a structured
**knowledge graph of math entities** (Definitions, Theorems, Problems), following
AutoMathKG's model (arXiv:2505.13406). This doc is the pick-up point for the next
session.

**Working branch:** `claude/design-plan-review-7zs2ii` (all work below is pushed there).

---

## TL;DR status

- The pipeline runs **end-to-end, no GPU**: PDF → `document.md` + `entities.json` (a sparse
  overlay of typed entities with member roles).
- The **extraction front-end is Mistral OCR + a vision correction pass** — chosen and
  validated this session over the previous local-docling + whole-page-vision-OCR front-end
  (now removed). Mistral does layout/reading-order/figure-extraction server-side; a Qwen3-VL
  correction pass proofreads each page against its image to catch Mistral's occasional subtle
  math errors.
- The **entity layer** (local typing + Stage-2 role attribution) is built and validated
  end-to-end on two real books. This is everything through AutoMathKG "Stage 1 + Stage 2."
- The **graph tier** (relationships/edges, MathVD fusion, LLM completion) is **not started** —
  that is the big next piece.
- The codebase was simplified to the minimal Mistral-only path (12 modules); the docling
  front-end and the (unused) DSPy optimizer subsystem were removed. 27 unit tests pass; they
  stub the heavy deps so they run anywhere.

---

## Architecture

One straight LangGraph map-reduce pipeline; every stage is `dispatch → worker → collect`
(see `pipeline.py`). Two phases split at the seam merger.

```
Phase 1 — per-page ingestion (backbone = `segments`, one per page)
  mistral_ocr → corrector → extractor

Phase 2 — flat node stream (backbone = `nodes`, global ordered list)
  seam_merger → problem_refiner → instruction_governor → entity_grouper → entity_attributor
  → assemble
```

- **`mistral_ocr`** calls Mistral's document OCR API (`mistral-ocr-latest`). Per page it
  returns reading-ordered markdown with figures referenced inline, plus each figure as a
  cropped image with a bbox. It builds the `Segment` backbone (`content` + `pictures`),
  rewrites Mistral's figure ids to the pipeline's positional `![N]()` convention, and renders
  each page to a `Segment.png` (via `pypdfium2`) so the corrector has an image to check.
- **`corrector`** proofreads every page's transcription against its image with a vision model
  (Qwen3-VL-235B). It fixes genuine transcription errors — especially math — and leaves
  faithful text untouched. A divergence guard rejects any "correction" whose length swings
  outside ±30% of the original (a runaway rewrite / truncation), keeping the original instead.
- **The seam merger births the flat stream.** It heals nodes split across page breaks, then
  flattens `segments[].nodes` into one global `nodes` list, stamping each `ASTNode` with a
  stable `id` and its originating `seg_index` (see `state.flatten_segments`). Everything after
  works on `nodes` keyed by id; pages survive only as `seg_index` for picture resolution at
  assembly.

### Module map (`src/module/`)

| file | role |
|---|---|
| `state.py` | data model + `State` (LangGraph channels), `flatten_segments`, `load_dspy_image` |
| `mistral_ocr.py` | **front-end**: Mistral OCR API → `Segment` backbone (markdown + figures + page renders) |
| `corrector.py` | **correction pass**: vision model proofreads each page vs its image; divergence-guarded |
| `extractor.py` | markdown → flat structural nodes (paragraph/math/list/header/…, plus `problem`/`instruction`) |
| `seam_merger.py` | heal page-split nodes; **birth the flat `nodes` list** |
| `problem_refiner.py` | tag each `problem` node with its `number` |
| `instruction_governor.py` | attach a shared lead `instruction` to the problems it governs |
| `entity_grouper.py` | **gather** Def/Thm/Example entities over windows + reconcile; wrap atomic problems 1:1 |
| `entity_attributor.py` | **Stage 2**: assign member roles (statement/proof/solution); lift number/instruction |
| `assembler.py` | walk `nodes` → `document.md`, resolving `![N]()` via `seg_index` |
| `pipeline.py` | graph wiring + `run()`; also writes `entities.json` |
| `llm.py` | `text_lm` (DeepSeek, text stages), `corrector_lm` (Qwen3-VL via OpenRouter, correction pass) |

### Entity data model (`state.py`)

- **3 types** (`EntityType`): `Definition`, `Theorem` (**subsumes** proposition/corollary/
  lemma), `Problem` (worked examples **and** atomic exercises).
- `Entity = {id, type, members: list[Member], number, instruction, …transient flags}`,
  where `Member = {node_id, role}` and `EntityRole ∈ {statement, proof, solution}`.
- The entity list is a **sparse overlay**: most nodes (prose, figures, headers) belong to
  no entity. There is no catch-all type.

---

## Key design decisions (and why)

**Front-end (decided this session, evidence below):**

1. **Mistral OCR is the front-end, not local docling.** The old front-end rendered/cropped
   pages locally with docling+torch (needs a GPU) and asked a vision LLM to transcribe a whole
   page and *infer* reading order. Mistral does layout, reading order, and figure extraction
   server-side over an API — no GPU — at parity on text/tables/reading-order and *better* on
   figure extraction (it returns cropped figures with bboxes, placed inline). This is the
   right fit for a GPU-less machine and collapses render + crop + OCR + figure-placement into
   one call.
2. **A correction pass, because Mistral makes occasional subtle math errors.** Measured:
   Mistral misreads things like a plain radical `√` as an indexed root `∛`, or attaches a
   subscript to the wrong symbol (`f(x)_1` for `f(x_1)`). These survive at any input
   resolution (model-inherent, not resolution-starved) and would silently corrupt the graph.
   A **generate-then-verify** second pass — a strong vision model re-reading the page image
   alongside the transcription — reliably fixes them; verification is an easier, safer task
   than transcribing from scratch, so the checker stays off its own OCR failure modes.
3. **Corrector = Qwen3-VL-235B, always-rewrite, every page, divergence-guarded.** The 235B
   model corrected both blatant and subtle errors with zero damage to correct pages; smaller
   models (8B/30B-a3b) were viable as a cheap *router* but had lower recall on subtle errors,
   so they were not put in the default path. "Always-rewrite over every page" was chosen over
   a math-only gate or a conditional-output router for simplicity and robustness; the divergence
   guard is the safety net. A cheaper conditional-output ("emit a sentinel when clean") variant
   is a drop-in future optimization behind the same interface. `CORRECTOR_MODEL` swaps the model.
   The corrector also **normalizes math delimiters** to the pipeline's `$`/`$$` convention so all
   downstream stages see uniform math: a deterministic token swap (`\[\]`→`$$`, `\(\)`→`$`, in
   `_normalize_math_delimiters`) runs on every page regardless of whether the vision correction was
   accepted, and the prompt asks the model to wrap any display equation the OCR left undelimited
   (bare `\begin{array}`/aligned/cases). Validated on the Oohama two-column paper (174 raw
   delimiters → 0, all array blocks fenced).
4. **The docling-geometry re-architecture was considered and rejected.** The original plan was
   to use docling's layout geometry to drive segmentation and kill the "duplication on complex
   layouts" artifact. A measurement (below) showed that artifact is **not systemic** — 0
   duplication across 11 adversarial pages — so the re-plumbing was not justified and the
   branch was dropped.

**Entity layer (unchanged from prior work):**

5. **Flat node stream as the post-extraction backbone**, born from the seam merger; pages
   demoted to a `seg_index` field. Stable node **ids** (not positions) because the entity
   overlay outlives the stage that made it and downstream stages delete nodes.
6. **Typing lives only at the entity layer.** The extractor stays structural. Grouping-cohesion
   happens where it's reliable: **atomic exercises stay cohesive at the extractor** (one node →
   wrapped 1:1); **worked Examples and Def/Thm are unbounded spans gathered at the entity
   layer**. Split by *structure*, not by type. **The extractor parses each page in isolation — no
   neighbour context.** Feeding neighbour pages as context made the LLM bleed their content into
   the current page's node list (measured ~25% duplicate entities; see Validation). Cross-page
   continuations are healed by the seam merger and shared instruction leads are attached
   positionally by the instruction governor, so the extractor needs only its own page.
7. **Dumb-greedy windows** over the flat stream; the LLM emits window-local positions and code
   resolves them to stable ids. **Reconciler = single-threaded post-pass** over the ordered
   entity list, merging `tail_open` ↔ `head_continuation` spans across windows.
8. **Role split is mechanical when possible** (`entity_attributor._marker_index` finds the
   explicit `Proof`/`Solution` boundary, heading or run-in); the LLM is only a fallback for
   markerless multi-member entities.

Full rationale is in the commit messages (`git log`) — this session's front-end work is in
`4d85caa`, `2cb75cd`, `4fb8ffd`, `63a288c`.

---

## Validation (real runs, this session)

All measured with the docling-free path on rendered pages; the scorer compared each page's
output to the rendered page image (ground truth). Corpus: **11 deliberately-hard pages** from
OpenStax Calculus Vol 1 (boxed theorems, worked examples, tables, dense exercises) and Judson
Abstract Algebra (run-in proofs, matrix/permutation exercises). Both books are single-column,
so this stresses boxed/floated reading order and math fidelity — **not** true multi-column.

- **Reading order / duplication:** essentially **0** on both Mistral and the old whole-page
  vision OCR across all 11 pages — this is what killed the docling-geometry re-architecture.
- **Mistral vs the old vision OCR:** parity on text, tables, and most math; Mistral wins figure
  extraction (the old GPU-less path could not extract figures at all). Mistral made subtle math
  errors the baseline did not — higher render resolution did **not** fix them.
- **Correction pass QA (production corrector over all 11 pages):** it changed **5** pages,
  and **every change was a correct fix** — including `\sqrt[3]→\sqrt`, `f(x)_1→f(x_1)`, and a
  `|x|→⌊x⌋` misread that *both* OCRs had shared (invisible to a text-only diff). It left the
  other 6 pages **byte-identical** (zero damage), and the divergence guard never misfired.
  Net: Mistral's real error rate on these adversarial pages was ~5/11, and the corrector caught
  all of them with no false positives.

Prior entity-layer validation (still valid): OpenStax continuity § — 13 entities, types 13/13,
roles 13/13; Judson polynomials § — 9 entities, types 9/9 (theorem-subsumption confirmed:
Corollary/Proposition/Lemma → Theorem), roles 9/9.

### Broadening run — Hefferon *Linear Algebra*, Ch.3 §III.2–§IV.2 (this session)

First **new-book** run since the front-end switch (Next-steps #1). Twelve pages (0-based
228–239) covering definitions, theorems, a Lemma + Corollary, run-in proofs, worked Examples,
and two dense Exercises sets — all three entity types and all three roles, matrix-heavy math.
End-to-end in ~155s → `document.md` + 63 entities. Two takeaways:

- **Good:** OCR reading order, matrix/LaTeX fidelity, and theorem-subsumption held; the corrector
  left content clean (no false rewrites, guard never misfired). Types and role splits looked right.
- **Defect found and fixed — extractor context bleed.** The first run produced ~25% duplicate
  entities: **13 extra Problem entities** (duplicated numbers 1.8, 2.12–2.22, 2.16 tripled) plus a
  malformed Theorem (`id=60`, two statements + four proofs = a duplicated statement/proof pair).
  Root cause, isolated by measurement:
  - **Bisected stage-by-stage:** page-local marker counts were `(1,1)` after OCR and after the
    corrector, jumped to `(2,2)` after the **extractor**, and persisted through seam-merge/flatten.
  - **Mechanism = context bleed, not self-repeat.** The extractor was fed each segment's neighbours
    as `previous_node_context` / `next_node_context`. Across 44 extractions **with** context there
    were **0 self-repeat** events but frequent neighbour bleed (a segment emitting a neighbour's
    content as its own nodes); the seam merger only heals boundary splits and never dedupes, so the
    bleed flowed into the entity layer. Since self-repeat never happens, a segment can only emit a
    neighbour's *content* via those context inputs.
  - **Fix:** the extractor now parses each segment **in isolation — no neighbour context** (see
    Key design decisions #6). Cross-page continuations are already healed by the seam merger and
    shared instruction leads are attached positionally by the instruction governor, so the context
    was only advisory. Re-run on the same 12 pages: **63→53 entities, duplicate problem numbers
    12→0**, the malformed Theorem gone, content markers `2→1`, and ~3× faster (155s→45s).
  - **Note on the earlier "duplication ≈ 0" result:** that measured the *OCR front-end's*
    reading-order/duplication (Mistral vs the old vision OCR) and still holds — Mistral's page
    markdown was clean. This defect was at a *different stage* (the extractor LLM's context inputs),
    which the 11-page front-end study never exercised.

### Multi-column run — Oohama, *On Two Strong Converse Theorems for DMCs* (this session)

Closes the long-standing **true multi-column** gap. A genuinely two-column IEEEtran
information-theory paper (arXiv:1008.1140, 5 pages) — dense Theorem/Lemma/Property/Proof
structure and heavy min/max display math, with strict column-major reading order (whole
left column, then whole right). End-to-end → `document.md` + 15 entities. Verified against
the page images:

- **Reading order: perfect.** On every page Mistral read the entire left column then the
  entire right column with **no cross-column interleaving** — e.g. page 3: Property 2 →
  Theorem 1 → Property 3 → Proof (all left), then eqs (3)/(4) → Theorem 2 → Theorem 3 → §4
  (all right), in exactly that order. This is the artifact the docling-geometry re-architecture
  feared; Mistral handles it server-side.
- **No duplication** (the extractor context-bleed fix holds on dense two-column pages).
- **Entity typing correct.** 13 theorem-type entities (Property 1–3 + Theorem 1–4 + Lemma 1–6,
  all subsumed to Theorem) + 2 definition clusters, **0 spurious Problems** (correct — a paper
  has no exercises). Statement/proof role split worked (4 entities carry proof members).
- **Two minor issues (not blockers):**
  1. *Coarse grouping* — the entity grouper bundles a run of adjacent `define X` display-math
     blocks into one multi-member Definition (6 statement members), and a Property + its long
     proof becomes one entity with ~10 members (one member per structural node). Defensible but
     coarse; revisit grouping granularity if downstream needs finer entities.
  2. *Inconsistent math delimiters* — **found and fixed.** Complex multi-line display equations
     arrived as raw `\[ … \]` / `\( … \)` / bare `\begin{array}` from OCR while simpler blocks used
     `$$`. The **corrector now normalizes delimiters** (see Key design decisions #3): a
     deterministic swap (`\[\]`→`$$`, `\(\)`→`$`) runs on every page, and the vision prompt wraps
     any display equation the OCR left undelimited. Re-run on the same paper: **174 raw delimiters
     → 0**, all `\begin{array}` blocks wrapped in `$$`, fences balanced. Side benefit: cleaner
     display blocks let the grouper separate the `define X` cluster (definitions 2→8), softening
     issue #1 above.

---

## Environment & how to run

**Three API keys** (in `.env` — see `.env.example` — or environment secrets):
- `MISTRAL_API_KEY` — page OCR (the front-end).
- `OPENROUTER_API_KEY` — the correction pass (Qwen3-VL-235B; `CORRECTOR_MODEL` /
  `CORRECTOR_PROVIDER` override the model / pinned upstream).
- `DEEPSEEK_API_KEY` — all text stages (extractor, seam, refiner, governor, entity
  grouping/attribution; `deepseek-v4-flash`).

**Deps** (uv) — **no GPU anywhere**:
- `uv sync` — light CPU core (dspy, langgraph, pydantic, neo4j, dotenv, httpx).
- `uv sync --extra mistral` — adds `pypdfium2` + `pillow`, used only to render page images for
  the correction pass.

**Tests:** `PYTHONPATH=src uv run pytest -q` (27 tests). `tests/conftest.py` stubs
dspy/pydantic/langgraph *only if absent*, so the suite runs with or without the real deps.

**Run the pipeline:**
```bash
PYTHONPATH=src uv run python -m module.pipeline book.pdf out/
# or, from Python, to limit pages (0-based):
PYTHONPATH=src uv run python -c "import asyncio; from module.pipeline import run; \
    asyncio.run(run('book.pdf', output_dir='out/', pages=[190,191,192]))"
# -> out/document.md, out/entities.json
```

---

## Known issues / limitations

- **Mistral's subtle math errors are real** (misread radicals/subscripts, model-inherent). The
  correction pass is the mitigation and tested clean on 11 pages, but that is an adversarial
  sample, not exhaustive — treat OCR'd math as verified-but-not-infallible.
- **The correction pass is a full Qwen3-VL-235B call per page.** Fine for correctness; if cost
  matters at scale, the documented conditional-output ("checker = router") variant skips the
  rewrite output on already-clean pages, and `CORRECTOR_MODEL` can point at a cheaper model.
- **Figure noise-filtering is not applied.** The old `image_filter` stage was docling-only and
  was removed; Mistral ignores decorative junk (covers/icons → 0 figures in testing) but a
  front-matter page can still yield thumbnail images — spot-check if you process front matter.
- **Multi-node proof/solution members are one member per structural node.** A long proof that
  spans several paragraphs/display-math blocks becomes several `proof` members on one entity (e.g.
  the Hefferon matrix-mult associativity theorem: statement + 7 proof nodes). This is correct span
  attribution, not duplication, but if a consumer wants a single proof blob it must join them.
- **Validation corpus:** 13 pages OpenStax/Judson + 12 pages Hefferon (single-column, three math
  books) **+ a 5-page true two-column IEEEtran paper (Oohama)**. Multi-column reading order is now
  covered and clean; the two-column sample is still small (one paper) — widen it before trusting
  heavily on multi-column, and note a *worked-example/exercise*-heavy two-column book is still
  untested (the Oohama paper has theorems/proofs but no Problems).

---

## Next steps (suggested order)

1. **Broaden extraction validation** — *in progress.* First new-book run (Hefferon Linear
   Algebra, this session) is done; it surfaced and **fixed** the extractor context-bleed defect
   (see Validation). Keep broadening: more sections/books, inspecting `document.md` *and* node
   structure (not just `entities.json`); watch for figure over-extraction on front matter and
   correction-pass regressions. **True multi-column reading order is now validated** (Oohama
   two-column paper, clean) — but on one small paper with no Problems; widen with a two-column
   *worked-example/exercise* book. Now that the extractor parses each page in isolation, also
   spot-check that classification quality at page boundaries did not regress (the removed context
   was advisory for boundary disambiguation; the seam merger should still stitch splits correctly).
   One follow-up remains from the two-column run: revisit **entity-grouping granularity** (long
   proofs become one member per node; some `define X` runs still bundle). (Display-math delimiter
   normalization — the other follow-up — is **done**, in the corrector.)
2. **Graph tier** (the big next piece) — relationship/edge discovery between entities
   (AutoMathKG's 9 tactic labels), then MathVD (embeddings/vector DB) for fusion and the
   Math-LLM completion step. `neo4j` is already a dep.

---

## Gotchas for the next session

- **Ephemeral container:** the scratchpad and any ad-hoc install do **not** survive a restart;
  only committed files do. New environment secrets are injected **at session start**, so a key
  added mid-session isn't visible until a fresh session.
- **Mistral key env-var name:** the hosted environment injects the Mistral secret as
  `MISTRAL_OCR_API`, but `.env.example` and the code's primary name is `MISTRAL_API_KEY`.
  `mistral_ocr._require_key` now reads `MISTRAL_API_KEY` first and **falls back to
  `MISTRAL_OCR_API`**, so it runs in either place; a local `.env` should still use
  `MISTRAL_API_KEY`. (`OPENROUTER_API_KEY` and `DEEPSEEK_API_KEY` match in both.)
- **Proxy port changes on restart:** outbound HTTPS goes through `$HTTPS_PROXY` (a
  `127.0.0.1:<port>` that changes when the worker restarts). httpx/litellm pick up the proxy +
  CA bundle from the environment (`SSL_CERT_FILE`), same for the Mistral, OpenRouter, and
  DeepSeek calls. A run launched before a restart fails with "Cannot connect to host
  127.0.0.1:<old-port>"; re-run from a fresh shell. Check `curl -sS "$HTTPS_PROXY/__agentproxy/status"`.
- **No GPU is needed anywhere** — the whole front-end is API-based now.
- **DeepSeek prompt caching** makes re-runs with unchanged prompts fast; changing a stage's
  prompt invalidates that stage's cache (expect a slower first re-run).
