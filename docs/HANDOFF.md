# KMS — Handoff

Knowledge-management pipeline that turns a math textbook PDF into a structured
**knowledge graph of math entities** (Definitions, Theorems, Problems), following
AutoMathKG's model (arXiv:2505.13406). This doc is the pick-up point for the next
session.

**Working branch:** `claude/entity-layer-math-design-00219i` (all work below is pushed there).

---

## TL;DR status

- The **entity layer is built and validated end-to-end on two real books.** A PDF →
  document.md + entities.json (a sparse overlay of typed entities with member roles).
- Everything through **AutoMathKG "Stage 1 + Stage 2"** is done: local typing + role
  attributes. The **graph tier** (relationships/edges, MathVD fusion, LLM completion)
  is **not started** — that's the big next piece.
- A **DSPy training-data + teacher-student optimizer** system exists (capture → trainsets
  → optimize) but is seeded thin and not yet wired into production.
- 25 unit tests pass; they stub the heavy deps so they run anywhere.

---

## Architecture

Two phases, split at the seam merger. LangGraph map-reduce; every stage is
`dispatch → worker → collect` (see `pipeline.py`).

```
Phase 1 — per-page ingestion (backbone = `segments`, one per page)
  image_filter → ocr → extractor

Phase 2 — flat node stream (backbone = `nodes`, global ordered list)
  seam_merger → problem_refiner → instruction_governor → entity_grouper → entity_attributor
  → assemble
```

- **The seam merger births the flat stream.** It heals nodes split across page breaks,
  then flattens `segments[].nodes` into one global `nodes` list, stamping each `ASTNode`
  with a stable `id` and its originating `seg_index` (see `state.flatten_segments`).
  Everything after works on `nodes` keyed by id; pages survive only as `seg_index` for
  picture resolution at assembly.

### Module map (`src/module/`)

| file | role |
|---|---|
| `state.py` | data model + `State` (LangGraph channels), `flatten_segments`, `load_segments` |
| `picture_extractor.py` / `docling_processor.py` | docling front-end: PDF → page PNGs + cropped figures (the `extract` extra) |
| `image_filter.py`, `ocr.py` | vision stages (OpenRouter): drop noise pictures, transcribe page → markdown |
| `extractor.py` | markdown → flat structural nodes (paragraph/math/list/header/…, plus `problem`/`instruction`) |
| `seam_merger.py` | heal page-split nodes; **birth the flat `nodes` list** |
| `problem_refiner.py` | tag each `problem` node with its `number` |
| `instruction_governor.py` | attach a shared lead `instruction` to the problems it governs |
| `entity_grouper.py` | **gather** Def/Thm/Example entities over windows + reconcile; wrap atomic problems 1:1 |
| `entity_attributor.py` | **Stage 2**: assign member roles (statement/proof/solution); lift number/instruction |
| `assembler.py` | walk `nodes` → `document.md`, resolving `![N]()` via `seg_index` |
| `pipeline.py` | graph wiring + `run()`; also writes `entities.json` |
| `llm.py` | `text_lm` (DeepSeek), `vision_lm` (OpenRouter), `teacher_lm` (optimization) |
| `capture.py`, `trainsets.py`, `optimize.py` | DSPy training-data + optimizer (see below) |

### Entity data model (`state.py`)

- **3 types** (`EntityType`): `Definition`, `Theorem` (**subsumes** proposition/corollary/
  lemma), `Problem` (worked examples **and** atomic exercises).
- `Entity = {id, type, members: list[Member], number, instruction, …transient flags}`,
  where `Member = {node_id, role}` and `EntityRole ∈ {statement, proof, solution}`.
- The entity list is a **sparse overlay**: most nodes (prose, figures, headers) belong to
  no entity. There is no catch-all type.

---

## Key design decisions (and why)

1. **Flat node stream as the post-extraction backbone**, born from the seam merger; pages
   demoted to a `seg_index` field. Rationale: three stages fought the per-page nesting;
   the entity layer would be the fourth. Stable node **ids** (not positions) because the
   entity overlay outlives the stage that made it and downstream stages delete nodes.
2. **Typing lives only at the entity layer.** The extractor stays structural. BUT
   grouping-cohesion happens where it's reliable: **atomic exercises stay cohesive at the
   extractor** (bounded mishmash, one node → wrapped 1:1); **worked Examples and Def/Thm
   are unbounded spans gathered at the entity layer**. Split by *structure*, not by type.
3. **Dumb-greedy windows** over the flat stream (whole nodes, never split a block; soft
   budget; three context budgets prev/cur/next). The LLM emits **window-local positions**;
   code resolves them to stable ids (LLM judges, code addresses — same as the governor).
4. **Reconciler = single-threaded post-pass** over the ordered entity list (not the seam
   merger's parity two-pass): where a `tail_open` entity meets a `head_continuation` one,
   merge; opener's type wins; chains across 3+ windows.
5. **Role split is mechanical when possible.** `entity_attributor._is_marker` finds the
   explicit `Proof`/`Solution` boundary — both a bare heading node (`### Solution`) **and**
   a run-in marker (`*Proof.* Suppose…`, Judson-style). Everything before is `statement`,
   the marker onward is proof/solution. The **LLM is only a fallback** for markerless
   multi-member entities (it was inconsistent on this split at temp 0).
6. **v1 = extraction only.** Roles/number/instruction are in; graph edges/fusion/completion
   are deferred to the graph tier.

Full rationale is in the commit messages (`git log`), especially `79fc3b9`, `89edf87`,
`187becb`, `1ee00ff`, `7b09ca8`, `f9c069a`.

---

## Validation (real runs)

Tested by rendering PDF pages → Segments tree → pipeline (docling bypass, see below).

- **OpenStax Calculus Vol 1** (continuity §): 13 entities — types 13/13, roles 13/13 after
  fixes. Worked Examples → Problems with statement/solution; checkpoints → 1:1 Problems.
- **Judson, Abstract Algebra** (polynomials §): 9 entities — types 9/9 (**theorem-
  subsumption confirmed**: Corollary/Proposition/Lemma all → Theorem), roles 9/9 after the
  run-in marker + OCR hardening.

Two extraction artifacts found and fixed via **prompt hardening** (both upstream of the
entity layer): a duplicated OpenStax checkpoint (reading-order/dedup) and a Judson
proposition whose statement+proof were fused into one node (proof-block separation).

---

## Environment & how to run

**Two API keys** (in `.env` — see `.env.example` — or environment secrets):
- `DEEPSEEK_API_KEY` — all text stages incl. entity grouping/attribution (`deepseek-v4-flash`).
- `OPENROUTER_API_KEY` — vision stages OCR/image_filter (`qwen3-vl-235b`).

**Deps** (uv):
- `uv sync` — light CPU core (dspy, langgraph, pydantic, neo4j, dotenv). Enough for the
  LLM pipeline + tests.
- `uv sync --extra extract` — adds docling[vlm] + CUDA torch (the PDF front-end). **Needs a
  GPU**; only for real docling extraction.

**On Claude Code on the web:** `.claude/hooks/session-start.sh` runs `uv sync` and persists
`.venv/bin` on PATH + `PYTHONPATH=src`, so `python`/`pytest`/`module` just work once it's
merged to the default branch. (The web container has **no GPU** and does **not** carry
`pypdfium2`/`pillow` — those are test-only, install ad-hoc.)

**Tests:** `PYTHONPATH=src uv run pytest -q` (25 tests). `tests/conftest.py` stubs
dspy/pydantic/langgraph *only if absent*, so the suite runs with or without the heavy deps.

**Run the full pipeline** (needs `--extra extract` + a GPU for docling):
```bash
PYTHONPATH=src uv run python -m module.pipeline book.pdf out/
```

**Run without a GPU (how all real testing was done):** bypass docling — render pages to a
Segments tree, then run with `extract_pictures=False`:
```bash
uv pip install pypdfium2 pillow                                   # test-only
PYTHONPATH=src uv run python scripts/pdf_to_segments.py book.pdf out/ --pages 190-193
PYTHONPATH=src uv run python -c "import asyncio; from module.pipeline import run; \
    asyncio.run(run('unused', output_dir='out/', extract_pictures=False))"
# -> out/document.md, out/entities.json
```
The vision LLM still does the real OCR on the page PNGs; only figure cropping is skipped
(so `![N]()` placeholders won't resolve — fine for text/entity testing). Test PDFs used:
OpenStax Calculus Vol 1 (`archive.org/download/CalculusVolume1LR/CalculusVolume1-LR.pdf`)
and Judson (`math1um.github.io/Teaching/judson-algebra.pdf`).

---

## DSPy training-data + optimizer

Groundwork for replacing hand-tuned prompts with data-driven optimization.

- **`capture.py`** — set `KMS_CAPTURE_DIR` and a run appends `{inputs, outputs}` JSONL per
  signature (no-op when unset). Extractor/grouper record at their LLM call; the attributor
  records at `collect` using the **final post-marker-split roles** → gold-quality labels.
- **`data/trainsets/`** — seed set from the two books: 8 extractor, 3 grouper, 12 attributor
  (silver labels; see its README). Loaded via `trainsets.load("<sig>")` → `dspy.Example`s.
- **`optimize.py`** — `BootstrapFewShot` teacher→student. `llm.teacher_lm()` is configurable
  via `TEACHER_MODEL` (defaults to the student → self-bootstrap). Per-signature exact-match
  metrics. Compiled programs → `data/compiled/` (gitignored). `load_into(predictor, name)`
  loads them into a stage's predictor.
  ```bash
  TEACHER_MODEL=<stronger-deepseek> PYTHONPATH=src \
      uv run python -m module.optimize entity_attributor entity_grouper
  ```
  Validated: teacher bootstrapped 4 metric-passing traces from the attributor trainset.

---

## Known issues / limitations

- **Trainset is a thin seed** (~3–12 per signature). Enough to bootstrap demos, thin for
  real optimization — needs more source pages through capture.
- **Optimizer not wired into production** — `load_into` is opt-in on purpose; don't
  auto-load a tiny-bootstrap program. Wire it in once a compiled program proves out on
  held-out pages.
- **Extractor optimization deferred** — its output is prose nodes; needs a fuzzier metric
  than exact-match.
- **Two upstream extraction edge cases remain possible** (both prompt-mitigated, not
  impossible): statement/proof fused into one OCR node with no marker; duplicated content
  on complex/boxed layouts.

---

## Next steps (suggested order)

1. **Graph tier** — the big unbuilt piece, and all LLM-only (no new infra): relationship/
   edge discovery between entities (AutoMathKG's 9 tactic labels), then later MathVD
   (embeddings/vector DB) for fusion and the Math-LLM completion step. `neo4j` is already a
   dep. Start with edge discovery over the `entities` list.
2. **Grow the trainset** — a few more sections/books through `KMS_CAPTURE_DIR`, then compile
   with a real `TEACHER_MODEL` and evaluate on held-out pages; wire `load_into` in behind a
   flag if it wins.
3. **Harden the extractor front-end** against the two edge cases if they recur at scale.

---

## Gotchas for the next session

- **Ephemeral container:** the scratchpad and any ad-hoc `uv pip install` (pypdfium2/pillow)
  do **not** survive a restart; only committed files do. `uv sync` on restart prunes non-
  pyproject packages.
- **Proxy port changes on restart:** outbound HTTPS goes through `$HTTPS_PROXY` (a
  `127.0.0.1:<port>` that changes when the worker restarts). A run launched before a restart
  will fail with "Cannot connect to host 127.0.0.1:<old-port>"; just re-run from a fresh
  shell. Check `curl -sS "$HTTPS_PROXY/__agentproxy/status"`.
- **No GPU on web** → use the docling bypass above for real-PDF tests.
- **DeepSeek prompt caching** makes re-runs with unchanged prompts fast; changing a stage's
  prompt invalidates that stage's cache (expect a slower first re-run).
- Keys and the SessionStart hook are set on the "Main" cloud environment.
