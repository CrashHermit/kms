# KMS — Handoff

Knowledge-management pipeline that turns a math textbook PDF into a structured
**knowledge graph of math entities** (Definitions, Theorems, Problems), following
AutoMathKG's model (arXiv:2505.13406). This doc is the pick-up point for the next
session.

**Working branch:** `claude/problem-finder-extractor-structure-neffix`. The entity-layer
redesign described below (pure-structural extractor + three per-type finders + flat entity
overlay with node provenance) is committed and pushed there, and validated end-to-end on
real Hefferon pages this session.

---

## TL;DR status

- The pipeline runs **end-to-end, no GPU**: PDF → `document.md` + `entities.json` +
  `nodes.json`. Validated this session on real pages (see Validation).
- The **extraction front-end is Mistral OCR + a vision correction pass** (Qwen3-VL). Mistral
  does layout/reading-order/figure-extraction server-side; the corrector proofreads each page
  against its image to catch Mistral's occasional subtle math errors and normalizes math
  delimiters. Unchanged this session; still solid.
- The **extractor is now purely structural** — it emits only general document structure
  (paragraph/math/list/table/image/caption/header/code) and knows nothing math-specific.
- The **entity layer is three independent per-type finders** (`problem_finder`,
  `definition_finder`, `theorem_finder`). Each is a self-contained copy of the same
  cursor-walk; they run in parallel and each emits a sparse overlay of its type. Validated:
  all three fire and produce coherent, non-overlapping entities.
- **Per-attribute passes** (member roles statement/proof/solution, `number`, `instruction`,
  title/field/refs) are **not built** — an entity is just `{id, type, members}` for now.
- The **graph tier** (relationships/edges, MathVD fusion, LLM completion) is **not started** —
  the big next piece.
- 26 unit tests pass; `conftest` stubs the heavy deps so they run anywhere.

---

## Architecture

One straight LangGraph pipeline (see `pipeline.py`). Two phases split at the seam merger.
The ingestion stages use the map-reduce `dispatch → worker → collect` shape; the three
finders are plain sequential nodes.

```
Phase 1 — per-page ingestion (backbone = `segments`, one per page)
  mistral_ocr → corrector → extractor

Phase 2 — flat node stream (backbone = `nodes`, global ordered list)
  seam_merger → { problem_finder, definition_finder, theorem_finder }  (parallel)
  then, after the graph: assemble + write nodes.json / entities.json
```

- **`mistral_ocr`** calls Mistral's document OCR API (`mistral-ocr-latest`). Per page it
  returns reading-ordered markdown with figures referenced inline, plus each figure as a
  cropped image with a bbox. It builds the `Segment` backbone, rewrites figure ids to the
  pipeline's positional `![N]()` convention, renders each page to a `Segment.png` (via
  `pypdfium2`) for the corrector, and sends `extract_header`/`extract_footer` so running
  heads / page numbers are split out of the page markdown.
- **`corrector`** proofreads every page's transcription against its image with Qwen3-VL-235B.
  It fixes genuine transcription errors (especially math) and leaves faithful text untouched;
  a ±30% length divergence guard rejects runaway rewrites. It also **normalizes math
  delimiters** to `$`/`$$` deterministically on every page.
- **`extractor`** parses each page's markdown into a flat list of **structural** nodes
  (paragraph/math/code/list/table/image/caption/header). Purely structural and
  domain-agnostic — no math-semantic typing, no problem/instruction node types, no
  proof/solution splitting. Parses each page in isolation (no neighbour context; passing
  neighbours caused ~25% duplicate-entity context bleed — see Key design decisions).
- **`seam_merger`** heals nodes split across page breaks (structural healing only — a block
  cut mid-way, judged on structure, not subject matter), then **flattens** `segments[].nodes`
  into one global `nodes` list, stamping each `ASTNode` with a stable `id` (document-order
  int) and its `seg_index`. Everything after works on `nodes` keyed by id.
- **The three finders** each cursor-walk `nodes` and emit a sparse overlay of one entity type
  (see Entity layer below). They run in parallel and each writes its own state channel.
- **After the graph:** `run()` concatenates the three overlays into one flat, document-ordered
  `entities.json` (assigning global ids), and writes `nodes.json` (the node stream, for
  provenance — an entity's `members` are node ids into it). `assemble` walks `nodes` →
  `document.md`, resolving `![N]()` via `seg_index`.

### Module map (`src/module/`)

| file | role |
|---|---|
| `state.py` | data model + `State` (LangGraph channels), `flatten_segments`, `load_dspy_image` |
| `mistral_ocr.py` | **front-end**: Mistral OCR API → `Segment` backbone (markdown + figures + page renders) |
| `corrector.py` | **correction pass**: vision model proofreads each page vs its image; divergence-guarded; delimiter normalization |
| `extractor.py` | markdown → flat **structural** nodes (no math typing) |
| `seam_merger.py` | heal page-split nodes (structural); **birth the flat `nodes` list** |
| `problem_finder.py` | **finder**: cursor-walk → Problem entities (worked examples AND exercises) |
| `definition_finder.py` | **finder**: cursor-walk → Definition entities |
| `theorem_finder.py` | **finder**: cursor-walk → Theorem entities (subsumes prop/cor/lemma; includes proof) |
| `assembler.py` | walk `nodes` → `document.md`, resolving `![N]()` via `seg_index` |
| `pipeline.py` | graph wiring + `run()`; flattens the 3 overlays and writes `entities.json` + `nodes.json` |
| `llm.py` | `text_lm` (DeepSeek, text stages), `corrector_lm` (Qwen3-VL via OpenRouter) |

### Entity data model (`state.py`)

- **3 types** (`EntityType`): `Definition`, `Theorem` (**subsumes** proposition/corollary/
  lemma), `Problem` (worked examples **and** exercises — AutoMathKG's model: same type,
  different place in the text).
- `Entity = {id, type, members}` where `members` is a `list[int]` of node ids (pointers back
  to the source nodes). No roles/number/instruction yet — those are future per-attribute passes.
- The overlay is **sparse**: most nodes (prose, figures, headers) belong to no entity.
- Overlays from the three finders are **independent** — they may reference the same node from
  more than one entity. That is fine because members are pointers, not copies; no merging or
  overlap arbitration is done.

### The finder shape (all three identical)

A cursor moves along the flat node stream. From the cursor it takes a look-ahead window of
whole nodes up to a soft token budget (`LOOKAHEAD_BUDGET=2000`); the LLM returns the entities
of that finder's type as `[start, end]` position spans. Advance is **structural, no
LLM self-report**: an entity is banked only once a node is seen to follow it (bounded); if the
only entity reaches the window edge it may continue, so the window **grows** (doubles, capped
at `MAX_LOOKAHEAD_BUDGET=8000`) and re-reads — never rewinds, no size guard, nothing truncated.
Each finder emits `Entity(type=…, members=[node ids])` in document order. The prompt tells the
model to START each span at the entity's OWN label node ("1.5 Theorem", "Example 6.7") and stop
at its boundary; theorem spans include the proof, problem spans include a shown solution.

The three finders are **deliberate copies**, not a shared abstraction (decided this session —
keep them dumb and explicit). If you change the walk, change it in three places.

---

## Key design decisions (and why)

**Front-end (unchanged, still valid):**

1. **Mistral OCR is the front-end, not local docling.** Mistral does layout, reading order,
   and figure extraction server-side over an API — no GPU — at parity on text/tables/reading
   order and better on figures (returns cropped figures with bboxes, placed inline).
2. **A correction pass, because Mistral makes occasional subtle math errors** (misreads `√`
   as `∛`, attaches a subscript to the wrong symbol). A generate-then-verify second pass with
   a strong vision model reliably fixes them; verification is easier/safer than transcribing
   from scratch. Qwen3-VL-235B, always-rewrite, every page, ±30% divergence-guarded. It also
   normalizes math delimiters to `$`/`$$`.

**Entity layer (this session's redesign):**

3. **Typing lives only at the entity layer; the extractor stays purely structural.** Makes the
   extractor domain-agnostic (reusable beyond math) and honours "typing lives at the entity
   layer." The extractor emits general document structure only — no `problem`/`instruction`
   node types, no proof/solution splitting (that was math-semantic knowledge leaking in).
4. **Three per-type finders, each its own focused cursor-walk**, replacing the old windowed
   `entity_grouper` + reconciler + Stage-2 `entity_attributor`. One prompt per type is far more
   reliable than one prompt juggling all types + roles. Each finder anchors on a type's own
   label cue and gathers its extent.
5. **Overlaps are allowed; the overlays are not merged.** Entities are node-id pointers, so two
   entities referencing the same node is harmless. We tried an arbitration/combiner step and
   removed it — no need. The three finders write three channels; `run()` concatenates them into
   one flat list for output.
6. **The extractor parses each page in isolation — no neighbour context.** Feeding neighbour
   pages made the LLM bleed their content into the current page (~25% duplicate entities,
   measured previously). Seam merger heals cross-page splits, so the context was only advisory.
7. **The seam merger heals purely on structure** — "is this block cut mid-way and continued?"
   — with no subject-matter reasoning (it used to reason about problem numbers; removed).
8. **Deliberate triplication over a shared finder core** (decided this session). The walk is
   ~90 lines; three copies with per-type prompts are easier to read and tune independently than
   one parameterized abstraction. Revisit only if they start drifting for no reason.

### Deferred decisions (recorded for the graph tier)

- **UUIDs vs ints for ids.** Today a node's `id` is a document-order int (identity + order in
  one). Harmless now because the node stream is immutable after flatten. At the **graph
  boundary**, mint a `uuid` as the stable vertex key (int ids collide across books; MathVD
  fusion needs global identity) and demote the int to an `index`/`order` provenance attribute.
  Entities need only a uuid (their order is derivable from `members[0]`). A reference graph does
  **not** preserve reading order for free — keep the node `index` as provenance if you want it.

---

## Validation (real runs, this session — Hefferon *Linear Algebra*, Ch.3 §III.1)

End-to-end, live (Mistral + Qwen3-VL + DeepSeek), no GPU. Both runs produced valid
`document.md` + flat `entities.json` + `nodes.json`.

- **Exposition pages 223–227 (5 pp, ~121s) — all three finders fire correctly.** 74 nodes →
  8 entities: **2 definitions** (1.2, 1.6), **1 theorem** (1.5, statement + proof span), **5
  problems** (worked Examples 1.4, 1.8–1.11, each a coherent multi-node span). Every entity
  starts at its own label node; **no cross-type overlaps**; connective prose correctly excluded.
- **Exercises pages 228–230 (3 pp, ~188s) — problem finder found all 19 exercises** but
  surfaced a real limitation: **node granularity is coarser than entity granularity for
  exercise lists.** The purely-structural extractor packs a run of exercises (e.g. 1.23, 1.24,
  1.25) into ONE `list` node, so the finder correctly identifies several problems inside it but
  they all collapse to the same `members=[node]` — indistinguishable pointers, and duplicate
  entities. This is NOT a finder bug (the finder behaves correctly given the node stream); it is
  a structural-vs-semantic granularity mismatch. See Known issues.

---

## Known issues / limitations

- **Exercise-list granularity (found this session).** When the extractor emits multiple
  exercises as one `list` node, per-problem entities can't be separated by node-id pointers.
  Options for later: (a) a splitting pass that subdivides multi-problem nodes; (b) sub-node
  span pointers (char ranges); (c) accept "one exercise list = one Problem entity"; (d) let the
  extractor split list items into per-item nodes (mildly less "pure structural"). Worked
  examples (exposition) don't have this problem — they're already multi-node spans.
- **No per-attribute detail yet.** Entities carry no roles, numbers, or instructions.
- **Mistral's subtle math errors are real**; the corrector is the mitigation, tested clean but
  on an adversarial sample, not exhaustive.
- **Validation corpus is still small for the finders** — Hefferon §III.1 (this session) plus
  the front-end's earlier multi-book corpus. Widen to more books/sections and inspect
  `document.md` + `nodes.json` + `entities.json` together.

---

## Environment & how to run

**Three API keys** (in `.env` — see `.env.example` — or environment secrets):
- `MISTRAL_API_KEY` — page OCR (the hosted env injects it as `MISTRAL_OCR_API`; the code reads
  `MISTRAL_API_KEY` first and **falls back to `MISTRAL_OCR_API`**).
- `OPENROUTER_API_KEY` — the correction pass (Qwen3-VL-235B; `CORRECTOR_MODEL` /
  `CORRECTOR_PROVIDER` override).
- `DEEPSEEK_API_KEY` — all text stages (extractor, seam, the three finders; `deepseek-v4-flash`).

**Deps** (uv) — **no GPU anywhere**:
- `uv sync` — light CPU core.
- `uv sync --extra mistral` — adds `pypdfium2` + `pillow` (render page images for the corrector).

**Tests:** `PYTHONPATH=src uv run pytest -q` (26 tests). `tests/conftest.py` stubs
dspy/pydantic/langgraph *only if absent*, so the suite runs with or without the real deps.

**Run the pipeline:**
```bash
PYTHONPATH=src uv run python -m module.pipeline book.pdf out/
# or, from Python, to limit pages (0-based):
PYTHONPATH=src uv run python -c "import asyncio; from module.pipeline import run; \
    asyncio.run(run('book.pdf', output_dir='out/', pages=[223,224,225]))"
# -> out/document.md, out/entities.json (flat [{id,type,members}]), out/nodes.json (provenance)
```
Good test PDF: Hefferon Linear Algebra — `https://jheffero.w3.uvm.edu/linearalgebra/book.pdf`
(525 pp; §III.1 exposition ≈ 0-based pages 223–227, exercises ≈ 228–230).

---

## Next steps (suggested order)

1. **Decide the exercise-list granularity** (Known issues #1) — it blocks clean per-exercise
   entities. Pick an option and implement; re-run 228–230 to confirm distinct problems.
2. **Per-attribute passes** — mirror AutoMathKG Step-2 / Appendix C: member roles
   (statement/proof/solution), `number`, `instruction`, then title/field/bodylist/refs. Each is
   its own LLM prompt over an entity's members (+ context for `instruction`). This is where the
   retired `problem_refiner`/`instruction_governor`/`entity_attributor` logic comes back, per
   attribute rather than bundled.
3. **Graph tier** (the big piece) — relationship/edge discovery between entities (AutoMathKG's
   9 tactic labels), then MathVD (embeddings/vector DB) for fusion and the Math-LLM completion
   step. Mint UUIDs here (see Deferred decisions). `neo4j` is already a dep.
4. **Broaden finder validation** — more books/sections, watching finder boundaries, figure
   over-extraction on front matter, and correction-pass regressions.

---

## Gotchas for the next session

- **Ephemeral container:** only committed files survive a restart. New environment secrets are
  injected **at session start**, so a key added mid-session isn't visible until a fresh session.
- **Mistral key env-var name:** hosted env injects `MISTRAL_OCR_API`; code falls back to it.
- **Proxy port changes on restart:** outbound HTTPS goes through `$HTTPS_PROXY`
  (`127.0.0.1:<port>` that changes on worker restart). A run launched before a restart fails
  with "Cannot connect to host 127.0.0.1:<old-port>"; re-run from a fresh shell. Check
  `curl -sS "$HTTPS_PROXY/__agentproxy/status"`.
- **No GPU is needed anywhere** — the whole front-end is API-based.
- **DeepSeek prompt caching** makes re-runs with unchanged prompts fast; changing a stage's
  prompt invalidates that stage's cache (slower first re-run).
- **The three finders are copies on purpose** — fix walk bugs in all three.
