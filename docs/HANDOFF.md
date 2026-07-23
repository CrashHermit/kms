# KMS — Handoff

Knowledge-management pipeline that turns a math textbook PDF into a structured
**knowledge graph of math entities** (Definitions, Theorems, Problems), following
AutoMathKG's model (arXiv:2505.13406). This doc is the pick-up point for the next
session.

**Working branch:** `claude/graph-dedup-planning-nwj0p7` (PR #12). The entity layer (pure-structural
extractor + three per-type finders + per-type attributors + the exercise **splitter** and the
**instruction distributor**) plus the **graph tier's structural provenance layer** (Neo4j `:Source`
+ `:Node` stream) are committed and pushed there, validated end-to-end. See the 2026-07-23 session
update for the graph work.

---

## TL;DR status

- The pipeline runs **end-to-end, no GPU**: PDF → `document.md`, and (when Neo4j is configured)
  the persisted `:Node` provenance layer + `:Entity` overlay in the graph. The graph now owns
  persistence — the old `entities.json`/`nodes.json` artifacts are gone. Validated on real pages
  (see Validation).
- The **extraction front-end is Mistral OCR + a vision correction pass** (Qwen3-VL). Mistral
  does layout/reading-order/figure-extraction server-side; the corrector proofreads each page
  against its image to catch Mistral's occasional subtle math errors and normalizes math
  delimiters. Solid.
- The **extractor is purely structural** — it emits only general document structure
  (paragraph/math/list/table/image/caption/header/code) and knows nothing math-specific.
- **The exercise splitter** (`entity/splitter.py`) runs between the seam merger and the
  finders. It rewrites the canonical node stream so each exercise is its OWN node (fixing the
  granularity mismatch below); an embedded lead-in is broken out onto its own node too.
- **The instruction finder** (`entity/instruction_finder.py`) runs immediately after the splitter
  and tags every exercise lead-in node `role="instruction"` — one uniform pass over the now-atomic
  stream (standalone and formerly-embedded lead-ins alike), consumed later by the distributor.
- The **entity layer is three independent per-type finders** (`problem_finder`,
  `definition_finder`, `theorem_finder`), each a self-contained copy of the same cursor-walk,
  running in parallel and emitting a sparse overlay of its type.
- **Per-type attributors are built and wired** (`entity/attributors/{problem,definition,theorem}.py`).
  Each enriches its finder's entities with the self-contained AutoMathKG attributes: label,
  number, title, field, contents, bodylist (Def/Thm), proofs (Thm), solutions (Prob).
- **The instruction distributor** (`entity/instruction_distributor.py`) runs at the end of the problem
  chain: a growing-window walk that copies a grouped-exercise lead-in's shared directive onto
  the `Problem.instruction` of the problems it governs (LLM-judged extent, no number matching).
- **Graph tier — structural provenance layer built** (this session): the flat node stream is
  persisted to Neo4j as a `:Source` node per book rooting its `:Node` markdown chain
  (`:HEAD`/`:NEXT`), reusing `core.NodeType`, wired in after the splitter, validated against a
  real Neo4j. **Still deferred:** the semantic tiers (dedup canonicals, general entities,
  concepts), cross-entity `refs`/`references_tactics`, MathVD fusion, and Math-LLM completion.
- **46 unit tests pass**; `conftest` stubs the heavy deps so they run anywhere.
- **This session (see Session update):** splitter + distributor stress-tested across 6
  committed fixtures (elementary→graduate) — both strong; base-instruction fixes to the splitter
  lead-in test and the attributor `number`; a **DSPy training harness** (`training/`, teacher
  `deepseek-v4-pro`, LLM-judge) built and proven but with no headroom to ship yet; and **ruff**
  adopted for format + lint (line-length 100).

---

## Session update — graph tier: structural provenance layer (2026-07-23)

Started the **graph tier** (phase 3, Neo4j) after planning the AutoMathKG + AutoSchemaKG hybrid for
the eventual semantic layer. This session built only the piece we're sure of — the **structural
provenance layer** — and left the semantic tiers as a design (below).

**What shipped (`src/kms/graph/`, all committed to `claude/graph-dedup-planning-nwj0p7`, PR #12):**
- `db.py` — the async Neo4j driver, the **only** module that imports `neo4j` (quarantined so the
  rest of the tier stays pure/unit-testable). Mirrors `core.llm`'s pattern (guarded dotenv, env-key
  constants, raise-on-use, a singleton) but uses an explicit module singleton + `close_driver`
  because a connection pool needs teardown, unlike the stateless LM config. Config: `NEO4J_URI` /
  `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE`; `is_configured()` gates the pipeline.
- `nodes.py` — pure `ASTNode → Neo4j` mapping, reusing `core.NodeType` (invents no node types).
  Deterministic `uuid5(source, index)` identity; each node carries the base `:Node` label **plus**
  its per-type label (`:Node:Math`, …); a `:Source` root per book with deterministic `source_uuid`.
- `entities.py` — pure `Entity → Neo4j` mapping, reusing `core.EntityType` (invents no types).
  Deterministic `uuid5(source, "entity#"+id)` identity (disjoint from node uuids); each entity carries
  the base `:Entity` label **plus** its per-type label (`:Entity:Theorem`, …). Scalar attributes +
  `contents` (string array) are native properties; the nested `bodylist`/`proofs`/`solutions` (the
  step material the future event layer will reify) are preserved as JSON-string properties for now.
- `schema.py` — idempotent bootstrap: uuid uniqueness constraints on `:Node`/`:Source`/`:Entity` +
  `source` indexes on `:Node`/`:Entity`. No vector index (embeddings belong to the semantic tiers).
- `writer.py` — `persist_nodes` (MERGE the `:Source` root + batched multi-label nodes, then wire
  `(:Source)-[:HEAD]->` first node and the `:NEXT` reading-order chain) and `persist_entities` (batched
  multi-label `:Entity` MERGEs, rooted under the `:Source` via `:HAS_ENTITY` and linked to member
  `:Node` s via `:DERIVED_FROM`). Pure planning (`node_batches`/`next_pairs`/`head_uuid`,
  `entity_batches`/`member_pairs`) is factored out and unit-tested.
- `persister.py` — two stages. `NodePersisterNode` is wired **after the splitter, before the finders**
  (the splitter re-ids the stream at `splitter.py:249`, so this is the first point the ids are final
  and match the entity `members`); `EntityPersisterNode` is the **fan-in after all three chains**, so
  it sees the fully attributed overlay, flattens it (`core.flatten_entities`), and writes it. Both are
  **no-ops** when Neo4j isn't configured or the run has no `source`, so DB-less runs and the test suite
  still complete (producing only `document.md`).
- `pipeline.py` — `run()` gained `source` (defaults to the PDF filename) + `title`/`author`, threaded
  via `State.source` / `State.source_metadata`; closes the driver in a `finally`. After the graph
  returns it only assembles `document.md` — persistence is entirely the graph tier's job now.

**Design decisions (planning, for the semantic tiers — NOT yet built):** a four-tier hybrid — the
math-specialized **mention** (Def/Thm/Prob, today's entities) → non-destructive **canonical** dedup
representative (synthesized, not elected) → AutoSchemaKG **general entity** (the connective hub that
decouples the reference graph from corpus completeness) → **concept** (abstract category). Dedup =
embed → (rerank) → LLM-judge, keeping instances distinct (AutoSchemaKG style) with AutoMathKG's
fusion mechanism. References route through the general-entity hub with the 9 tactic labels; Neo4j is
the durable "Existing KG", each run an "Input KG" fused in. Events tier: likely dropped for math.

**Validation.** 78 hermetic tests pass (neo4j stubbed in `conftest`); ruff clean. The opt-in
integration test (`tests/test_graph_db_integration.py`) is gated on an explicit `KMS_NEO4J_IT=1`
flag — NOT on `NEO4J_URI`, because a configured `.env` (which `db.py` loads) would otherwise pull
the slow live tests into every `pytest`. It was run against a **real Neo4j 5.26** — schema
bootstrap, multi-label nodes, the `:Source`/`:HEAD`/`:NEXT` graph with `title`/`author`, all
idempotent on re-run. **Aura note:** Bolt (7687) is blocked from the web-session sandbox
(HTTPS-proxy-only network policy). `db.py` now carries a second transport for exactly this: set
`NEO4J_TRANSPORT=http` and the tier talks to the SAME Aura instance over its HTTPS Query API
(`POST /db/<db>/query/v2`, port 443) with the SAME creds — the https endpoint is derived from the
`neo4j+s://` host, so nothing else changes. `HttpQueryDriver`/`HttpQuerySession` mirror the tiny
`session().run()` slice the tier uses (httpx client, honours `HTTPS_PROXY`/`SSL_CERT_FILE`), so the
whole graph stack runs unchanged from the sandbox. Verified end-to-end from the web session: the
Query API is reachable on 443 and the full `driver → verify_connectivity → ensure_schema` path lands
on Aura (the last live check stopped only at auth — the `.env` password was stale, not a transport
issue). `.env` (gitignored) holds the creds. Leave `NEO4J_TRANSPORT` unset for normal Bolt.

---

## Session update — stress test + DSPy harness + code conventions (2026-07-22)

**Stress-test fixtures (`tests/fixtures/books/`, committed).** Six small PDF page-slices of
openly-licensed books (CC BY / BY-SA family), so splitter/distributor stress tests reuse them
without re-downloading full books. See that dir's `README.md` for provenance/licences/levels.
Coverage spans the exercise-governance spectrum: OpenStax *Elementary Algebra 2e*
(`ea2e_ch1_review` = dense short "In the following exercises" lead-ins + interleaved headers;
`ea2e_sec1_3_exercises` = a section set + an "Everyday Math" block with per-exercise embedded
directives); OpenStax *Calculus Vol 3* (`calc3_gradients_exercises` = "For the following
exercises" phrasing, advanced, `[T]` tech exercises); Lebl *Basic Analysis* I & II
(`lebl_realanalysis_sec2_1`, `lebl2_metricspaces_sec8_1` = graduate, self-contained
"Exercise N.N.N:" proof exercises, ~no shared lead-ins); Hefferon *Linear Algebra*
(`hefferon_linsys_exercises` = per-exercise imperatives + "✓/X" recommended glyph — the
own-numbered-exercise-is-not-a-lead-in case).

**Findings — splitter + distributor are both strong across all styles.**
- Splitter: exact splits (112/112 and 74/74 exercises on the EA2e slices; 23/23 on Lebl I),
  precise lead-in tagging with **zero false positives** — range-less "In/For the following
  exercises", Hefferon per-exercise imperatives (0 tagged), Lebl proof exercises all handled.
- Distributor: governed runs bounded correctly; word-problems with embedded directives left
  ungoverned; no over-extension across intervening lead-ins/headers.
- **Real defects found (reported; mostly NOT splitter/distributor):**
  - *Problem attributor number* is format-sensitive — misses bare multi-column numbers
    ("925.") and once grabbed an in-text cross-reference ("2.1.12 Prove Proposition 2.1.13" →
    number 2.1.13). **Base-instruction fix applied** (anchor on the leading own-number, never an
    in-text reference).
  - *Front-end OCR* drops short interstitial lead-in lines on dense 3-column pages (calc3: ~4 of
    ~6 "For the following exercises" lead-ins never became nodes), so the distributor can't stamp
    them — a Mistral-OCR fidelity issue, not the governor.
  - *Distributor* soft over-extension: a "find the limit" lead-in stamped onto a following
    "finish the proof" exercise (Lebl 2.1.11).
- **Splitter base-instruction fix applied:** its lead-in TAG "decisive test" no longer hinges on
  naming an explicit range (range-less lead-ins are the common real case).

**DSPy training harness (`training/`) — data-driven stage tuning instead of prompt-chasing.**
Teacher `deepseek-v4-pro` bootstraps traces; a reference-free **LLM-as-judge** (also on the
teacher) filters them into few-shot demos for the flash student — DSPy *compilation*, not weight
fine-tuning (DeepSeek's API doesn't expose that). Pieces: `core/tracing.py` (opt-in
`KMS_TRACE_DIR` capture) and per-stage `training/<stage>/{metric,dataset,compile}.py`; the
splitter and distributor nodes auto-load `training/<stage>/compiled.json` if present (override
with `KMS_SPLITTER_PROGRAM` / `KMS_DISTRIBUTOR_PROGRAM`).
- **Cheap-data source:** distributor examples come from a live-captured `distributor.jsonl`
  (`KMS_TRACE_DIR` on a run) — `training/distributor/dataset.py::load_traces`. (An earlier
  reconstruct-from-`out/<fixture>/` path, 41 examples from 5 runs with zero new calls, retired with
  the `nodes.json`/`entities.json` artifacts when the graph became the store.)
- **Pilot results:** both stages already judged near-perfect, so naïve `BootstrapFewShot` had no
  headroom — splitter dev pass-rate 1.00→1.00 (and its demos are ~2.7k-token page-sized windows,
  a bad fit); distributor 0.90→0.80 (within noise; demos small but non-targeted). **Neither
  compiled program was shipped** — the bare students are already strong. The harness is the
  deliverable; the next lever is *targeted* hard-case demos (seed the over-extension cases)
  and/or MIPROv2 instruction-opt, measured on a larger reconstructed eval set.

**Code conventions — ruff adopted (was: no tooling at all).** `ruff` = the one style tool
(format + light lint), configured in `pyproject.toml`: line-length **100**, src-layout aware,
lint set `F/E/I/B/UP` (E501 off — the formatter owns wrapping and long DSPy `description=`
strings are intentional). The whole repo was reformatted in one isolated commit. It codifies the
existing de-facto house style: double quotes, modern typing (`X | None`, `list[...]` — no
`Optional`/`List`), stdlib→third-party→local import groups, rich module-level rationale
docstrings (`r"""` for DSPy signatures with LaTeX), no `from __future__` (runtime is 3.14).
**Before committing Python: `uv run ruff format . && uv run ruff check .`** (both must be clean).

---

## Architecture

One straight LangGraph pipeline (see `pipeline.py`). Two phases split at the seam merger.
The ingestion stages use the map-reduce `dispatch → worker → collect` shape; the three
finders are plain sequential nodes.

```
Phase 1 — per-page ingestion (backbone = `segments`, one per page)
  mistral_ocr → corrector → extractor

Phase 2 — flat node stream (backbone = `nodes`, global ordered list)
  seam_merger → splitter → instruction_finder
    → { problem_finder    → problem_attributor    → instruction_distributor }   (parallel
      { definition_finder → definition_attributor }                              chains,
      { theorem_finder    → theorem_attributor }                                 one per type)
    → entity_persister (fan-in: flatten overlays → :Entity graph)
  then, after the graph: assemble document.md
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
- **`splitter`** (`entity/splitter.py`) normalises the flat stream once, before the finders.
  Any node that packs two-or-more numbered exercises (the extractor emits a run of exercises as
  ONE `list` node) is **replaced in place by one node per exercise** (an embedded lead-in broken
  out onto its own node too). It re-ids the stream; split pieces inherit their parent's
  `seg_index`. This makes exercises atomic so the finders emit one clean entity each (see the
  granularity note in Key design decisions). See "The splitter" below.
- **`instruction_finder`** (`entity/instruction_finder.py`) runs right after the splitter: a
  cursor-walk that tags every exercise lead-in node `role="instruction"`. Because the splitter has
  already made every lead-in a standalone node, this is one uniform per-node decision (no
  split/segment work). Its output is tiny (a list of positions), so it can't truncate.
- **The three finders** each cursor-walk `nodes` and emit a sparse overlay of one entity type
  (see Entity layer below). They run in parallel and each writes its own state channel.
- **The three attributors** (`{problem,definition,theorem}_attributor.py`) each run after their
  finder, enriching that overlay's entities with the self-contained AutoMathKG attributes in
  place. See "The attributors" below.
- **`instruction_distributor`** runs after the problem attributor: a growing-window walk that
  stamps `Problem.instruction` from the instruction finder's tagged lead-in nodes. See "The
  instruction distributor" below.
- **`entity_persister`** is the fan-in of all three chains (the pipeline's terminal stage): it
  flattens the three overlays into one flat, document-ordered list (assigning global ids), then
  upserts them as the `:Entity` graph layer — each rooted under the book's `:Source` via
  `:HAS_ENTITY` and linked to its member `:Node` chunks via `:DERIVED_FROM`. A no-op when Neo4j isn't
  configured. See the graph tier section.
- **After the graph:** `run()` only assembles — `assemble` walks `nodes` → `document.md`, resolving
  `![N]()` via `seg_index`. All persistence happens inside the graph (node + entity persisters).

### Module map (`src/kms/`)

Organized by phase (see `ARCHITECTURE.md` for the dependency rule). Dependencies point backward
only: `core ← ingestion ← entity ← graph ← output`.

| file | role |
|---|---|
| `core/models.py` | data model (`ASTNode`/`Segment`/`Entity`/…, `EntityType`, `FIELDS`), `flatten_segments`, `flatten_entities`, `merge_results_into_segments` — dspy/langgraph-free |
| `core/state.py` | the LangGraph `State` (channels + reducers); imports `models` |
| `core/llm.py` | `text_lm` (DeepSeek, text stages), `corrector_lm` (Qwen3-VL via OpenRouter) |
| `core/tracing.py` | opt-in per-call trace capture (the data→compile loop's raw material) |
| `ingestion/ocr.py` | **front-end**: Mistral OCR API → `Segment` backbone (markdown + figures + page renders) |
| `ingestion/corrector.py` | **correction pass**: vision model proofreads each page vs its image; divergence-guarded; delimiter normalization |
| `ingestion/extractor.py` | markdown → flat **structural** nodes (no math typing) |
| `ingestion/seam_merger.py` | heal page-split nodes (structural); **birth the flat `nodes` list** |
| `entity/splitter.py` | **splitter**: split packed exercise nodes → one node per exercise; tag lead-ins `role="instruction"` |
| `entity/finders/problem.py` | **finder**: cursor-walk → Problem entities (worked examples AND exercises) |
| `entity/finders/definition.py` | **finder**: cursor-walk → Definition entities |
| `entity/finders/theorem.py` | **finder**: cursor-walk → Theorem entities (subsumes prop/cor/lemma; includes proof) |
| `entity/attributors/problem.py` | **attributor**: label/number/title/field/contents + solution split |
| `entity/attributors/definition.py` | **attributor**: label/number/title/field/contents + bodylist (4 roles) |
| `entity/attributors/theorem.py` | **attributor**: label/number/title/field/contents + bodylist + proofs |
| `entity/instruction_distributor.py` | **distributor**: growing-window; stamp `Problem.instruction` from tagged lead-ins |
| `output/assembler.py` | walk `nodes` → `document.md`, resolving `![N]()` via `seg_index` |
| `graph/entities.py` | `Entity → Neo4j` mapping (deterministic uuids, multi-label, nested attrs as JSON strings) |
| `graph/persister.py` | `NodePersisterNode` (after splitter) + `EntityPersisterNode` (fan-in): persist the two graph layers |
| `pipeline.py` | graph wiring + `run()`; after the graph, only assembles `document.md` (persistence is the graph tier's) |
| `cli.py` | `__main__` entry point: `python -m kms.cli book.pdf out/` |

### Entity data model (`core/models.py`)

- **3 types** (`EntityType`): `Definition`, `Theorem` (**subsumes** proposition/corollary/
  lemma), `Problem` (worked examples **and** exercises — AutoMathKG's model: same type,
  different place in the text).
- `Entity = {id, type, members, …attributes}`. `members` is a `list[int]` of node ids (pointers
  back to the source nodes); a finder emits just `{type, members}` and the type's attributor
  fills the rest: `label`, `number`, `title`, `field`, `contents`, `bodylist` (Def/Thm),
  `proofs` (Thm), `solutions` (Prob), plus `instruction` (Prob, filled by the distributor, not
  the attributor). Unset attributes are omitted when persisted, so a bare entity is just
  `{id, type, members}` → a minimal `:Entity` vertex. Cross-entity `refs`/`references_tactics` are
  **not** here — later graph tier.
- `ASTNode` now carries a `role` field — a non-structural annotation kept **off** the structural
  `NodeType`. The splitter sets `role="instruction"` on exercise lead-ins; the distributor reads
  it. It is written onto the `:Node` vertex only when set.
- The overlay is **sparse**: most nodes (prose, figures, headers) belong to no entity.
- Overlays from the three finders are **independent** — they may reference the same node from
  more than one entity. That is fine because members are pointers, not copies; no merging or
  overlap arbitration is done. (Because the splitter makes exercises atomic upstream, the
  problem finder no longer produces duplicate-membered exercise entities — see Key decisions.)

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

### The splitter (`entity/splitter.py`)

One LLM walk over the flat stream doing one local, per-node job (**SPLIT**), then a single
deterministic rebuild:

- For a node that packs ≥2 numbered exercises, the LLM returns each exercise's `number` +
  verbatim `content` (subparts kept nested, markers like `✓` kept). The rebuild replaces that
  node with one node per exercise (number as literal leading text). A leading **orphan fragment**
  (a previous exercise's continuation the OCR left at the head of the list node, e.g. trailing
  `(d)…(e)…`) and an **embedded lead-in** ("9-16 Sketch the polar curve." between the exercises)
  are each returned as an empty-number item and emitted as their own node, so nothing is dropped
  and every lead-in becomes a standalone node for the instruction finder to tag.
- **Content-based, not offset/anchor-based** — the LLM cannot reproduce LaTeX verbatim, so
  anchor-into-original slicing fails (tried, reverted); the fix for the orphan loss was a prompt
  change, not a mechanism change.

The splitter does **not** tag lead-ins — that was folded out into `instruction_finder` so each
stage has one focused prompt (the split prompt no longer juggles the "own number ≠ lead-in"
caveat), and tagging happens once, uniformly, over the atomic stream.

Decisions are per-node (a node either is or isn't a packed list) and a node's content is wholly
inside one window, so there is **no cross-window banking** — the walk gathers splits keyed by
node id, then rebuilds once and re-ids. Re-iding is safe because it runs before any entity
overlay exists (nothing references the old ids yet).

### The instruction finder (`entity/instruction_finder.py`)

A cursor-walk that runs right after the splitter and tags every exercise lead-in node
`role="instruction"`. Because the splitter has already made every lead-in a standalone node,
this is one uniform per-node decision — no split/segment work. The decisive test in the prompt: a
node beginning with its **own** number is an exercise, never a lead-in (this killed 8 false
positives on Hefferon, where the section has *zero* true lead-ins); range-less lead-ins ("In the
following exercises, …") are the common real case and are tagged too. Output is a tiny list of
positions, so — unlike the splitter's verbatim output — it can't hit output-token truncation.

### The attributors (`{problem,definition,theorem}_attributor.py`)

Each runs after its finder and fills the **self-contained** AutoMathKG attributes on that type's
entities, in place, reading only the entity's own member nodes (drawing `FIELDS`/`ACTIONS`
taxonomies from `core/models.py`):

- **Problem** — one identity pass (label/number/title/field + a `solution_start` boundary);
  members split into statement vs shown solution, both halves always kept. No bodylist (Table B3
  restricts it to Def/Thm). Deliberately does **not** fill `instruction` (that's the distributor).
- **Definition** — identity pass + a contents pass (label peeled off) + a bodylist pass over
  only the four roles a definition uses (premise/assumption/definition/enumeration).
- **Theorem** — identity + statement bodylist + a per-proof pass (each proof gets contents +
  bodylist); statement vs proof split on a boundary, like the problem's solution split.

### The instruction distributor (`entity/instruction_distributor.py`)

`instruction` is a cross-entity, positional attribute (a lead-in governs a *run* of problems and
is a member of none of them), so it is **not** a per-entity attributor pass — it is one more
stage on the problem chain, after the attributor. It is a **growing-window walk** (same shape as
the finders): anchor on a `role="instruction"` node, take a look-ahead window of the problems
that follow it (up to the next lead-in), and the **LLM judges which it governs — by meaning, not
numbers** — returning the governed run + the shared imperative. Grow the window while the run
reaches the edge; bank when a non-governed problem bounds it or the candidates are exhausted;
stamp `instruction` on the governed problems. There is **no range/number parsing** — that was an
earlier design and was replaced, because a lead-in like "Prove each of the following." has no
numbers and governance is semantic (the walk correctly stops at a following *compute* problem).

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
9. **Fix exercise granularity at the NODE level (splitter), not the entity level.** The problem
   was that the extractor packs a run of exercises into one `list` node, so the finder can only
   point duplicate entities at it. We first tried an **exercise governor** — a post-finder pass
   that split the list into fine entities and reconciled them against the finder's coarse ones —
   and **removed it**: it was unstable run-to-run (it oscillated between splitting one list node
   and enumerating the whole page, sometimes swallowing a worked example and letting
   reconciliation delete the finder's correct capture). The splitter attacks the root cause
   instead: once nodes are atomic, the finder/attributor "just work," no reconciliation, precise
   provenance. It sits at the structural→semantic boundary (allowed to know exercise
   conventions) so the **extractor stays purely structural**, and it runs **after** the seam
   merger because a cross-page exercise list is only whole there.
10. **Instruction distribution is a growing-window walk, LLM-judged — no range parser.** A
    lead-in's governed run is a semantic question (many lead-ins state no numeric range), so the
    LLM decides the extent by reading the following problems, in the same growing-window shape as
    the finders. A number/range-matching version was built first and replaced.
11. **Split content-based, harden by prompt.** The splitter's LLM returns each exercise's content
    (not offsets/anchors into the original) — the model can't reproduce LaTeX verbatim, so anchor
    slicing fails. Residual content loss (an orphaned continuation fragment) was fixed by a prompt
    instruction to emit the orphan as its own node, not by changing the mechanism.

### Deferred decisions (recorded for the graph tier)

- **UUIDs vs ints for ids — DONE for nodes** (this session). A node's `id` stays a document-order
  int in memory; at the graph boundary `graph.nodes.node_uuid` mints the stable vertex key as
  `uuid5(source, index)` — **deterministic**, so re-persisting a book MERGEs onto the same vertices
  instead of duplicating, and `source` disambiguates the same index across books. The int is
  demoted to an `index` provenance property, and reading order is kept both as `index` and as
  `:NEXT` edges. Entities get the same treatment: a deterministic `uuid5(source, "entity#"+id)`,
  where `id` is the document-order position from `flatten_entities` (order derives from `members[0]`).

---

## Validation (real runs, this session — Hefferon *Linear Algebra*, Ch.3 §III.1)

End-to-end, live (Mistral + Qwen3-VL + DeepSeek), no GPU. Both runs produced valid
`document.md` + flat `entities.json` + `nodes.json`.

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
  soft failure is a task-kind over-extension (Lebl 2.1.11). The `training/` harness targets
  exactly this decision if we want to push it further.
- **Problem attributor `number` is format-sensitive** — see Session update (bare multi-column
  numbers, in-text cross-references). Base-instruction fix applied; verify on the next runs.
- **Front-end drops short interstitial lead-in lines** on dense multi-column exercise pages
  (calc3) — an OCR/corrector fidelity gap upstream of the governor, not a splitter/distributor
  bug. Worth a targeted look at the corrector prompt or Mistral options for such layouts.
- **Mistral's subtle math errors are real**; the corrector is the mitigation, tested clean but
  on an adversarial sample, not exhaustive.
- **Validation corpus is still small** — Hefferon §III.1 plus the front-end's earlier multi-book
  corpus. Widen to more books/sections and inspect `document.md` alongside the persisted `:Node` +
  `:Entity` graph. Cross-entity attributes (`refs`/`references_tactics`) are still unbuilt (graph tier).

---

## Environment & how to run

**Three API keys** (in `.env` — see `.env.example` — or environment secrets):
- `MISTRAL_API_KEY` — page OCR (the hosted env injects it as `MISTRAL_OCR_API`; the code reads
  `MISTRAL_API_KEY` first and **falls back to `MISTRAL_OCR_API`**).
- `OPENROUTER_API_KEY` — the correction pass (Qwen3-VL-235B; `CORRECTOR_MODEL` /
  `CORRECTOR_PROVIDER` override).
- `DEEPSEEK_API_KEY` — all text stages: extractor, seam, splitter, the three finders +
  attributors, and the distributor (`deepseek-v4-flash`). Also powers the DSPy **teacher**
  (`deepseek-v4-pro`, compile-time only; `TEACHER_MODEL` overrides) — same key.

**Deps** (uv) — **no GPU anywhere**:
- `uv sync` — light CPU core.
- `uv sync --extra mistral` — adds `pypdfium2` + `pillow` (render page images for the corrector).

**Tests:** `PYTHONPATH=src uv run pytest -q` (105 tests). `tests/conftest.py` stubs
dspy/pydantic/langgraph *only if absent*, so the suite runs with or without the real deps.

**Style (ruff):** `uv run ruff format . && uv run ruff check .` — both must be clean before
committing Python. Config in `pyproject.toml` (line-length 100, lint set `F/E/I/B/UP`). See the
Session update for the codified conventions.

**Types (pyright): advisory-only, NOT a gate.** `pyright` is configured in `pyproject.toml` but
not enforced, and `PYTHONPATH=src uv run pyright src/kms` currently reports **19 known
pre-existing errors** — none are bugs. They're all benign: LangGraph `add_node`/`StateNode`
generic friction (workers typed `state: dict`), runtime-safe nullability the dispatch guards
already ensure (`X | None` at OCR/LLM boundaries used where non-`None` is guaranteed), the
optional `pypdfium2` import (the `mistral` extra), and `total=False` TypedDict item access. Don't
chase these to green piecemeal — it's churn on non-bugs. If pyright is ever made a real gate,
green it in one deliberate pass (`assert ... is not None` at the guarded points, type workers as
`State`, `# pyright: ignore` the optional import) and wire it into the workflow; until then treat
a *rising* count as the signal (a refactor that adds errors), not the absolute number.

**Stress-test the governor:** run the pipeline on a fixture and inspect, e.g.
`PYTHONPATH=src uv run --extra mistral python -m kms.cli
tests/fixtures/books/ea2e_ch1_review.pdf out/ea2e_ch1_review`. To capture DSPy training traces,
prefix with `KMS_TRACE_DIR=out/traces`. Compile a stage:
`PYTHONPATH=src uv run python -m training.distributor.compile out/<run_dir> ...`.

**Run the pipeline:**
```bash
PYTHONPATH=src uv run python -m kms.cli book.pdf out/
# or, from Python, to limit pages (0-based):
PYTHONPATH=src uv run python -c "import asyncio; from kms import run; \
    asyncio.run(run('book.pdf', output_dir='out/', pages=[223,224,225]))"
# -> out/document.md; with NEO4J_* set, also the persisted :Node + :Entity graph
```
Good test PDF: Hefferon Linear Algebra — `https://jheffero.w3.uvm.edu/linearalgebra/book.pdf`
(525 pp; §III.1 exposition ≈ 0-based pages 223–227, exercises ≈ 228–230).

---

## Next steps (suggested order)

1. **Extensive splitter + distributor validation — DONE this session** (see Session update): 6
   fixtures across elementary→graduate styles, both stages strong. Remaining follow-ups from it:
   the front-end lead-in-loss on dense multi-column pages (calc3), and — if we want to push
   quality past "already strong" — *targeted* hard-case demos or MIPROv2 via the `training/`
   harness, measured on a larger reconstructed eval set. The optimizer infra is built and proven;
   it just has little headroom on the current, already-good stages.
2. **Cross-entity attributes** — `refs` / `references_tactics` (AutoMathKG's 9 tactic labels
   between entities). These are inherently graph-tier (they relate entities), so they fold into
   the next item rather than being per-entity passes.
3. **Graph tier** (the big piece) — **structural provenance layer DONE this session** (`:Source`
   + `:Node` stream in Neo4j; see the 2026-07-23 session update). Remaining: the **semantic tiers**
   on top — dedup canonicals, general entities, concepts (the AutoMathKG + AutoSchemaKG hybrid),
   relationship/edge discovery with `refs`/`references_tactics`, then MathVD (embeddings/vector DB)
   for fusion and the Math-LLM completion step. Node UUIDs are already minted (see Deferred
   decisions); entities/canonicals need theirs when they land.
4. **Broaden front-end/finder validation** — more books/sections, watching finder boundaries,
   figure over-extraction on front matter, and correction-pass regressions.

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
- **The three finders (and three attributors) are copies on purpose** — fix walk bugs in all.
- **Run ruff before committing** (`uv run ruff format . && uv run ruff check .`); no `from
  __future__` (runtime is 3.14). The whole repo was reformatted once — that commit is isolated
  for `git blame`.
- **Reuse the committed fixtures** in `tests/fixtures/books/` for stress tests; don't re-download
  full books. Distributor training data comes from a live-captured `distributor.jsonl` — set
  `KMS_TRACE_DIR` on a run to grow the trainset (the old reconstruct-from-`out/<run>/` path retired
  with the JSON artifacts).
- **`uv run` re-syncs and drops the `mistral` extra.** A plain `uv run …` (e.g. `pytest`) after
  `uv sync --extra mistral` uninstalls `pypdfium2`/`pillow`, so the next pipeline run dies with
  "No module named 'pypdfium2'". For a full run use `uv run --extra mistral python …` (or re-sync
  the extra first). The test suite does not need it.
