# KMS — Handoff

Pipeline that turns a textbook PDF into a knowledge graph of pedagogical blocks, the procedures
that derive them, and their steps.

**Read this first, then `SCHEMA.md`** (what the graph is) and `REBUILD.md` (how it got here).
`ARCHITECTURE.md` covers the package layout and its dependency rules; `CONCEPT-LAYER.md` specs the
one tier that is still empty; `STYLE.md` is the Python style guide.

---

## TL;DR status

The pipeline runs end to end: PDF → OCR + vision correction → structural nodes → pedagogical
blocks + their procedures → Neo4j. No GPU anywhere; three hosted API keys.

**The AutoMathKG entity layer is gone.** It was replaced (see `REBUILD.md`) by one general chain:

```
group_finder → statement_extractor → procedure_extractor
```

where nine per-type modules (three finders, three attributors, three referencers) used to stand.
What changed, and why it matters:

- **The finder emits untyped spans in two roles**, `entity` and `procedure`. A theorem and its
  proof are two adjacent spans, so the statement/derivation boundary is now a **structural
  detection** rather than the old attributors' semantic `proof_start`/`solution_start` call.
- **`type` is open and induced** (`definition` / `theorem` / `example` / `law` / `mechanism` / …),
  filled by the statement extractor, stored as a property on a bare `:Entity`. The closed
  three-value enum and the per-type Neo4j labels are gone, so a physics or biology book types
  itself with no profile.
- **Decomposition is universal.** Every procedure decomposes into verbatim `:Act` steps. The old
  schema restricted its step list to Theorems and Definitions, which left **every solution
  stepless** — the procedural spine was empty for exactly the exercise-heavy books this pipeline
  targets.
- **The graph is four semantic kinds**: `:Entity`, `:Procedure`, `:Act`, `:Concept`, over the
  `:Source`/`:Node` provenance tier. Deleted outright: the mention/canonical split, `:REFERENCES`,
  `:USES`, `:REALIZES`, `:DEPENDS_ON`, `:BROADER`, and the closed `FIELDS` / `ACTIONS_ALL` /
  `REFERENCE_KINDS` vocabularies.

**One tier is dark.** The concept layer writes nothing: its only source was `Entity.field` from
AutoMathKG's field taxonomy. `graph/concepts.py` keeps the hub identity scheme (`normalize_concept`,
`concept_uuid`, `concept_properties`, `concept_batch`) for conceptualization to build on. Because
block-to-block relations are also gone, **a book's blocks currently connect to nothing outside
their own `:Source`** — closing that is `CONCEPT-LAYER.md`, and it is the highest-value next step.

**Re-validated live** (2026-07-25) — see `robustness_test/ENTITY-REBUILD-VALIDATION.md`. Five
fixture books ran end to end. The structural contracts hold exactly (0 dangling refs, 0 span
overlaps, 0 duplicate-member entities, 8/8 procedures an exact verbatim partition), theorem+proof
came out as two attached spans 4/4 on Stein, and the open `type` induces non-math genres (`law`,
`mechanism`) on a direct probe. Two behavioural gaps are open: **unmarked derivations produce no
procedure spans** (Lebl: 0 procedures despite three worked examples — deterministic across 3 runs,
so that book's procedural spine is empty), and **`type` keys off a block's embedded content**
(Hammack: 14/16 exercises typed theorem/example/quote). The graph *write* path is still unverified —
the Aura credential is stale (see "Known issues").

---

## Pipeline

```
ocr → corrector → extractor → seam_merger → splitter → instruction_finder
    → node_persister → group_finder → statement_extractor → procedure_extractor
    → instruction_distributor → entity_persister
```

Two phases split at the seam merger.

**Ingestion (backbone `segments`, map-reduce).** Mistral OCR turns each page into reading-ordered
markdown plus extracted figures. The corrector proofreads each page against its image (Qwen3-VL).
The extractor parses corrected markdown into purely **structural** nodes — no semantic typing. The
seam merger heals nodes split across page breaks, then flattens to one global ordered `nodes` list
with stable ids.

**Entity layer (backbone `nodes`, sequential).** The splitter rewrites any node packing several
exercises so each exercise is atomic. The instruction finder tags lead-in nodes
`role="instruction"`. The node persister writes the finalized stream as the `:Source`/`:Node`
provenance layer — after the splitter, so persisted ids match the overlay's members. Then:

- **`group_finder`** — one cursor-walk with a growing look-ahead window, kept verbatim from the old
  problem finder. It banks a span only once a node is seen to follow it (so a window cut can never
  split one), and grows the window when the sole span reaches the edge. Emits `entity` and
  `procedure` spans; **one partition** — a node belongs to at most one span.
- **`statement_extractor`** — one universal pass filling `type`, `label`, `number`, `title`,
  `contents`. It never restructures the span.
- **`procedure_extractor`** — decomposes each procedure span into verbatim ordered steps and
  attaches it to the **nearest preceding block**. Steps partition the content exactly.
- **`instruction_distributor`** — copies a lead-in's shared directive onto the blocks it governs.
- **`entity_persister`** — orders the overlay, writes `:Entity` then `:Procedure`/`:Act`.

Persistence is a no-op when Neo4j is unconfigured, so a DB-less run still produces `document.md`.

---

## Behaviours worth knowing

**Absence is structural.** No procedure span means no procedure — there is no "is there something
to work out?" call. This is the same principle that makes the finder's banking reliable.

**Orphan procedures are kept, not dropped.** A derivation with no preceding block (a proof deferred
pages after its theorem) is decomposed and returned unattached rather than discarded — its text and
steps are real extracted content. Resolving an explicit cross-reference ("Proof of Theorem 2.4") is
deliberately not implemented; the attachment rule is undecided.

**Persisted vs. transient.** The in-memory models are working state; the graph is the deliverable.
A field is persisted only if something reads it *from the graph*. `ASTNode.role` and all of
`Segment` are transient. This rule exists because its absence produced three dead properties
(`role`, `BodySegment.action`, `Entity.bodylist`) that were written to Neo4j and never read.

**One partition costs something.** With three independent finders, an inline definition embedded
mid-proof could be caught by the definition finder regardless of what else claimed those nodes.
With one partition it is absorbed into the containing block. **Expect definition counts to drop**
on a re-run versus the numbers recorded below — that is intended.

---

## Environment & how to run

**Three API keys** (in `.env` — see `.env.example` — or environment secrets):
- `MISTRAL_API_KEY` — page OCR (the hosted env injects it as `MISTRAL_OCR_API`; the code reads
  `MISTRAL_API_KEY` first and **falls back to `MISTRAL_OCR_API`**).
- `OPENROUTER_API_KEY` — the correction pass (Qwen3-VL-235B; `CORRECTOR_MODEL` /
  `CORRECTOR_PROVIDER` override).
- `DEEPSEEK_API_KEY` — all text stages: extractor, seam, splitter, instruction finder, group
  finder, both extractors, and the distributor (`deepseek-v4-flash`).

**Deps** (uv) — **no GPU anywhere**:
- `uv sync` — light CPU core.
- `uv sync --extra mistral` — adds `pypdfium2` + `pillow` (render page images for the corrector).

**Tests:** `PYTHONPATH=src uv run pytest -q` (145 tests, 3 skipped). `tests/conftest.py` stubs
dspy/pydantic/langgraph *only if absent*, so the suite runs with or without the real deps. The
Neo4j integration test is opt-in behind `KMS_NEO4J_IT=1`.

**Style (ruff):** `uv run ruff format . && uv run ruff check .` — both must be clean before
committing Python. Config in `pyproject.toml`; conventions in `docs/STYLE.md`.

**Types (pyright): advisory-only, NOT a gate.** `pyright` is configured in `pyproject.toml` but not
enforced, and it reports known pre-existing errors — none are bugs. They are LangGraph
`add_node`/`StateNode` generic friction (workers typed `state: dict`), runtime-safe nullability the
dispatch guards already ensure, the optional `pypdfium2` import, and `total=False` TypedDict item
access. Don't chase these to green piecemeal — it is churn on non-bugs. Treat a *rising* count as
the signal (a refactor that adds errors), not the absolute number.

**Run the pipeline:**
```bash
PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/
# or, from Python, to limit pages (0-based):
#   from kms import run; run(pdf, output_dir="out/", pages=[...])
# -> out/document.md; with NEO4J_* set, also the persisted :Node + :Entity + :Procedure/:Act graph

# every stage logs an INFO summary; KMS_LOG_LEVEL=DEBUG adds one line per DSPy call
KMS_LOG_LEVEL=DEBUG PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/
```

A healthy INFO run reads like this (Morris, *Topology Without Tears*, 4 pp) — the stage counts
are the fastest check that a run behaved:

```
kms.ingestion.extractor: extractor: 4 page(s) -> 38 node(s)
kms.ingestion.seam_merger: seam merger: 4 page(s) -> flat stream of 37 node(s)
kms.entity.splitter: splitter: 37 node(s) -> 38 (1 packed node(s) split)
kms.entity.instruction_finder: instruction finder: 38 node(s) -> 0 lead-in(s) tagged
kms.entity.group_finder: group finder: 38 nodes -> 10 block(s), 1 procedure span(s)
kms.entity.statement_extractor: statement extractor: 10 block(s) typed | exercise=6 definition=2 example=2
kms.entity.procedure_extractor: procedure extractor: 1 span(s) -> 1 attached, 0 orphan, 16 step(s)
```

**Fixture books** for stress runs (12 PDFs, do not re-download): `tests/fixtures/books/` and
`robustness_test/books/`. `robustness_test/REPORT.md` records the prior behaviour of each.

---

## Next steps (suggested order)

1. **Teach the group finder to cut unmarked derivations.** *(Was: re-run the fixture books — done
   2026-07-25, see `robustness_test/ENTITY-REBUILD-VALIDATION.md`.)* The finder only detects a
   procedure where the book marks one (`Proof.`, `Solution.`). Lebl marks nothing, so it yields 0
   procedure spans and its worked examples decompose into no steps at all — the same empty
   procedural spine the rebuild set out to remove, reached a different way. The cut is available:
   the derivation already sits in its own nodes. This is the drift risk item 3 predicts, and a
   worked example with no procedure is a countable regression metric.
2. **Build the concept layer** (`CONCEPT-LAYER.md`). It is the only connective tissue between
   blocks now, so until it lands the graph has no cross-book structure at all. Conceptualization is
   probe-validated and needs no corpus-global pass.
3. **Tune the group finder's Signature.** It now carries a harder job than the finder it replaces —
   a reliable structural task (find labeled block boundaries) merged with a softer one (find the
   statement→derivation turn in unmarked prose). The banking machinery is untouched and orthogonal;
   the drift risk is in the prompt. The `procedure`-follows-`entity` post-check is the cheap
   detector.
4. **Decide the orphan attachment rule**, then write orphan procedures to the graph.
5. **Fusion** (`CONCEPT-LAYER.md`) — independent track, needs the concept layer populated first.

---

## Validation (historical — Hefferon *Linear Algebra*, Ch.3 §III.1)

> **These numbers predate the entity-layer rebuild** and describe the three per-type chains. They
> are kept because they record what the front-end and the pedagogy stages (splitter, instruction
> finder, distributor) actually did on real pages — all of which are **unchanged**. Treat the
> entity counts as the baseline to diff against, not as current behaviour.

End-to-end, live (Mistral + Qwen3-VL + DeepSeek), no GPU. Both runs produced valid
`document.md` (plus, at the time, the flat `entities.json` / `nodes.json` artifacts — since
**retired**: persistence is now entirely the graph tier, so a current run writes only `document.md`
and, when Neo4j is configured, the graph).

- **Exposition pages 223–227 (5 pp, ~121s) — all three finders fire correctly.** 74 nodes →
  8 entities: **2 definitions** (1.2, 1.6), **1 theorem** (1.5, statement + proof span), **5
  problems** (worked Examples 1.4, 1.8–1.11, each a coherent multi-node span). Every entity
  starts at its own label node; **no cross-type overlaps**; connective prose correctly excluded.
- **Exercises pages 228–230 — the granularity mismatch, now SOLVED by the splitter.** The
  extractor packs a run of exercises (1.23…1.30) into ONE `list` node; the finder used to
  collapse them to duplicate `members=[node]` pointers. With the splitter in front, the full
  end-to-end run now yields **19 distinct Problem entities (numbers 1.12–1.30), zero
  duplicate-member groups** — each exercise atomic with precise provenance.

**Splitter (this session), live on the real exercises page, 3/3 runs consistent:** the
3518-char list packing 1.23–1.30 splits into 8 atomic nodes; node 9 (1.13/1.14) splits;
every number 1.13–1.30 heads exactly one node; **zero false lead-in tags** (this section has no
true lead-ins); content mass preserved **0.992** (residual is cosmetic whitespace). Note a
run-to-run *decision* sensitivity remains (see Known issues) — much smaller blast radius than
the retired governor, since a miss just leaves a coarse node rather than corrupting entities.

**Instruction distributor (this session), live on constructed lead-ins:** "In Exercises
1.23-1.25, …" governs 1.23–1.25 and correctly excludes a following *Prove* problem; **"Prove
each of the following." (no numbers) governs the two prove problems and excludes a following
*compute* problem** (the case a range parser can't do); "For each of the following …" governs
the whole run.

---

## Known issues / limitations

- **Splitter decision variance.** Whether the LLM splits a given packed list node still varies
  a little run-to-run (temp 0). Consistent on the Hefferon page across 3 runs, but not
  guaranteed elsewhere. A miss is *safe* (the node stays coarse — one entity for the list —
  rather than corrupting anything), but it under-splits. Candidate for a DSPy-optimised prompt
  (the splitter has a narrow, checkable contract, so it is the most trainable stage — this is
  where the "train itself" idea lands).
- **Splitter is near-lossless, not lossless.** ~0.8% residual mass on the test page is cosmetic
  whitespace from the content copy; the orphan-fragment loss is fixed. A truly lossless split
  would need offsets, which the model can't produce over LaTeX (see decision 11).
- **Instruction distributor — now validated on real lead-in-heavy sections** (see the Session
  update). Correct extent + bounding across OpenStax algebra/calc and Lebl analysis; the one
  soft failure is a task-kind over-extension (Lebl 2.1.11).
- **`number` extraction is format-sensitive** — bare multi-column numbers and in-text
  cross-references both mislead it. The guard ("the block's OWN leading number, never an in-text
  cross-reference", with worked counter-examples) carried into the statement extractor's Signature;
  verify on the next real runs.
- **Front-end drops short interstitial lead-in lines** on dense multi-column exercise pages
  (calc3) — an OCR/corrector fidelity gap upstream of the governor, not a splitter/distributor
  bug. Worth a targeted look at the corrector prompt or Mistral options for such layouts.
- **Mistral's subtle math errors are real**; the corrector is the mitigation, tested clean but
  on an adversarial sample, not exhaustive.
- **Validation corpus is still small** — Hefferon §III.1 plus the twelve fixture books. Widen to
  more books/sections and inspect `document.md` alongside the persisted `:Node` + `:Entity` +
  `:Procedure`/`:Act` graph.
- **Unmarked derivations are invisible to the group finder** — next step 1. Deterministic on Lebl
  (0 procedure spans, 3/3 runs).
- **`type` keys off a block's embedded content, not its nature.** Hammack: 14/16 exercises typed
  `theorem`/`example`/`quote`, because `statement_extractor` sees only the block's own member nodes
  and runs *before* `instruction_distributor`, which holds the governing lead-in. The run has the
  disambiguating evidence and types the blocks anyway. This is the baseline's old `field` gap #3
  moved onto `type`, where it matters much more.
- **The Neo4j credential is stale**, so the graph *write* path is unverified. Bolt is blocked as
  documented; the HTTP transport reaches Aura and returns `Unauthorized: Invalid credential`. Note
  `NEO4J_*` set-but-invalid is worse than unset — the persisters gate only on `NEO4J_URI` being
  present, so a run would crash rather than skip. The driver-free mapping layer was validated
  offline against real output instead (uuid disjointness, 0 dangling edge endpoints, `:Act` chain
  shape).
- **Trace capture is gone.** `core/tracing.py` and `KMS_TRACE_DIR` no longer exist after the
  rebuild, but `CLAUDE.md` and `robustness_test/REPORT.md` still describe them. No training data was
  captured for the live validation run; restore this before tuning any Signature.

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
- **DSPy's own disk cache (`~/.dspy_cache`) will silently serve a prior run's answers**, and
  it is on by default. This is a validation trap, not just a speed-up: re-running a stage on
  an unchanged input returns byte-identical output in a fraction of the time, which looks
  exactly like model determinism. It caught the 2026-07-25 sweep once (a "3/3 identical runs"
  claim that was really 3 cache hits in 4.5 s). To measure real behaviour, disable it:
  `dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)`. As a smell test,
  a genuine DeepSeek stage call is seconds, not milliseconds.
- **There is logging, but still no tracing.** Every stage logs one INFO line summarising what it
  produced, and `KMS_LOG_LEVEL=DEBUG` adds one line per DSPy call (inputs' shape + elided outputs,
  ~70 lines/book); loggers are module-named, so one stage can be turned up alone. This is a
  debugging aid only — `core/tracing.py` / `KMS_TRACE_DIR` are still gone, so a run still produces
  NO trainable per-call capture. For structured I/O you must keep the LangGraph `State` yourself or
  read `~/.dspy_cache` (responses only — the cache is keyed by a request hash, so inputs are gone).
- **There is one finder now, not three copies.** A walk bug is fixed in one place
  (`entity/group_finder.py`); the banking machinery there is load-bearing and was carried over
  verbatim — change the Signature freely, the cursor logic only with care.
- **Run ruff before committing** (`uv run ruff format . && uv run ruff check .`); no `from
  __future__` (runtime is 3.14). The whole repo was reformatted once — that commit is isolated
  for `git blame`.
- **Reuse the committed fixtures** in `tests/fixtures/books/` for stress tests; don't re-download
  full books.
- **`uv run` re-syncs and drops the `mistral` extra.** A plain `uv run …` (e.g. `pytest`) after
  `uv sync --extra mistral` uninstalls `pypdfium2`/`pillow`, so the next pipeline run dies with
  "No module named 'pypdfium2'". For a full run use `uv run --extra mistral python …` (or re-sync
  the extra first). The test suite does not need it.
