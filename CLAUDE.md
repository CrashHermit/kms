# KMS

Pipeline that turns a math textbook PDF into a knowledge graph of math entities
(Definitions, Theorems, Problems), following AutoMathKG (arXiv:2505.13406).

**Read `docs/HANDOFF.md` first** — it has the architecture, design decisions and their
rationale, validation results, full run instructions, and the container gotchas. This file
is just the quick reference.

## Current focus

The extraction front-end is **Mistral OCR + a vision correction pass** (no GPU), validated
on adversarial pages. The entity layer is **built**: an exercise **splitter** makes exercises
atomic at the node level, an **instruction finder** then tags exercise lead-in nodes
`role="instruction"`, three per-type **finders** (problem/definition/theorem) each build a
sparse overlay, three per-type **attributors** fill the self-contained AutoMathKG attributes,
and an **instruction distributor** propagates a grouped-exercise lead-in's directive onto the
Problems it governs. Three per-type **referencers** then extract each entity's cross-entity `refs`
(the definitions/theorems it cites, each with a tactic label). The **graph tier** (Neo4j) is now the
pipeline's persistence layer, and the AutoMathKG + AutoSchemaKG **unified substrate, math-first** (see
`docs/UNIFIED-KG.md`). It has **five built, wired layers**: the structural **provenance** layer — a
`:Source` node per book rooting its `:Node` markdown stream via `:HEAD`/`:NEXT` edges (reusing
`core.NodeType`) — the **`:Entity:Mention` overlay** on top of it — one Definition/Theorem/Problem
vertex per entity, rooted under its `:Source` via `:HAS_ENTITY` and linked to its member `:Node`
chunks via `:DERIVED_FROM`, carrying the self-contained AutoMathKG attributes — the **procedural
layer** — each Theorem `proof` / Problem `solution` reified into a `:Procedure` (hung off the entity
via `:HAS_PROCEDURE`) rooting a `:Event` step chain (`:FIRST`/`:THEN`) — the **concept layer** — a
global, born-canonical `:Concept` per entity `field`, joined by an `:INSTANCE_OF` edge — and the
**reference layer** — `:REFERENCES` edges (tactic on the edge) from an entity onto a global
`:Entity:Canonical` **hub** per referenced target (so citations from any entity/book converge on one
node), plus step-level `:USES {tactic}` edges from a proof `:Event` onto those same canonicals, and
`:REALIZES` edges tying each canonical **back** to the in-corpus Definition/Theorem mention that
realizes it (nominal title-match) — so a citation resolves *through* the shared hub to where the
concept is actually defined, which is the cross-corpus convergence payoff. All are wired into the
entity persister and covered by the opt-in integration test (`KMS_NEO4J_IT`, runnable against a live
instance; the `:Node` layer was also validated end-to-end against a real Neo4j). The graph **owns
persistence** (the old `entities.json`/`nodes.json` artifacts are gone). Still to come (see
`docs/UNIFIED-KG.md`): richer per-entity concepts + the `:BROADER` concept taxonomy (MSC-anchored), the
`:DEMONSTRATES`/`:PRACTICES` anchor edges, the **semantic** dedup that refines `:REALIZES` (MathVD
embedding fusion), Math-LLM completion, and the whole generalization layer.

## Layout

- `src/kms/` — the pipeline, organized by phase (see `docs/ARCHITECTURE.md` for the full
  rationale and the backward-only dependency rule). Packages:
  - `core/` — shared center that every stage depends on and that depends on no stage:
    `models.py` (domain data, dspy/langgraph-free), `state.py` (the LangGraph `State`),
    `llm.py` (LM config), `tracing.py`.
  - `ingestion/` — phase 1 (backbone `segments`): `ocr.py` (Mistral front-end), `corrector.py`,
    `extractor.py` (purely structural), `seam_merger.py`. Map-reduce `dispatch → worker → collect`.
  - `entity/` — phase 2 (backbone `nodes`): `splitter.py`, `instruction_finder.py`,
    `finders/{problem,definition,theorem}.py`, `attributors/{problem,definition,theorem}.py`,
    `referencers/{problem,definition,theorem}.py`, and `instruction_distributor.py` (problem chain
    only). Plain sequential nodes.
  - `output/` — `assembler.py` (runs after the graph).
  - `graph/` — phase 3 (Neo4j). **Provenance + `:Entity` overlay + procedural + concept + reference
    layers built** (the unified-KG substrate, math-first — see `docs/UNIFIED-KG.md`): `db.py` (async
    driver, the only neo4j import; plus an `NEO4J_TRANSPORT=http` HTTPS Query-API transport for
    sandboxes where Bolt/7687 is blocked), `nodes.py` (ASTNode→Neo4j), `entities.py` (Entity→Neo4j,
    `:Entity:Mention` + per-type label), `procedures.py` (proofs/solutions→`:Procedure`/`:Event`),
    `concepts.py` (`field`→global `:Concept` + `:INSTANCE_OF`), `references.py` (refs→`:Entity:Canonical`
    hubs + `:REFERENCES` edges), `uses.py` (step-level `:Event`→`:Canonical` `:USES` edges) and
    `realizes.py` (`:Mention`→`:Canonical` `:REALIZES` identity edges, nominal title-match) — all
    deterministic uuids, driver-free; `schema.py` (constraint/index bootstrap for all layers),
    `writer.py` (`persist_nodes` + `persist_entities` + `persist_procedures` + `persist_concepts` +
    `persist_references` + `persist_uses` + `persist_realizes`), `persister.py` (the two pipeline stages:
    `NodePersisterNode`, `EntityPersisterNode` — the latter persists the entity, procedural, concept,
    reference, step-level `:USES` and `:REALIZES` layers in order). Not started: richer concepts +
    `:BROADER` (MSC), the `:DEMONSTRATES`/`:PRACTICES` anchors, the semantic dedup that refines
    `:REALIZES` (MathVD fusion), Math-LLM completion, and the generalization layer.
  - `pipeline.py` wires the graph; `cli.py` is the `__main__` entry; `kms/__init__.py` exposes `run`.
- Flow: `ocr → corrector → extractor → seam_merger → splitter → instruction_finder →
  node_persister → {problem,definition,theorem} finder → {…} attributor → {…} referencer →
  entity_persister`, and the problem chain has one more stage, the instruction distributor, before the
  fan-in. Two phases split at `seam_merger`: per-page ingestion (backbone `segments`) → flat global
  node stream (backbone `nodes`, stable ids). The **splitter** rewrites `nodes` so each exercise (and
  each embedded lead-in) is its own node; the **instruction finder** then tags every lead-in node
  `role="instruction"` over that atomic stream. The three finders walk `nodes` in parallel and write
  their own entity channel; each attributor enriches its channel in place (overlap is fine — members
  are node-id pointers); each referencer then extracts that channel's cross-entity `refs`. The
  `node_persister` stage (after the splitter, before the finders) writes the node stream to Neo4j as
  the `:Source`/`:Node` provenance layer; the `entity_persister` fan-in stage (after all three chains)
  flattens the overlays into one document-ordered, globally-id'd list, writes them as the `:Entity`
  overlay, then the procedural (`:Procedure`/`:Event`), concept (`:Concept`/`:INSTANCE_OF`), reference
  (`:REFERENCES` edges onto `:Entity:Canonical` hubs), step-level `:USES`, and `:REALIZES`
  (mention→canonical identity) layers on top. All
  persist only when Neo4j is configured (`NEO4J_*` env vars) and are no-ops otherwise, so a DB-less run
  still produces `document.md` but persists nothing. The finders, attributors, and referencers are
  self-contained copies of one shape.
- `docs/HANDOFF.md` — full context. `docs/ARCHITECTURE.md` — the package layout and its rules.

## Commands

- Deps: `uv sync` (light CPU core) · `uv sync --extra mistral` (adds `pypdfium2` + `pillow`,
  used to render page images for the correction pass). **No GPU anywhere.**
- Tests: `PYTHONPATH=src uv run pytest -q` (154 tests) — `conftest` stubs the heavy deps, so it
  runs anywhere, no keys needed.
- Run (full pipeline): `PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/`,
  or from Python `from kms import run; run(pdf, output_dir="out/", pages=[...])` to limit pages
  (0-based). Writes `out/document.md` and, when Neo4j is configured (`NEO4J_*`), persists the
  `:Node` + `:Entity` graph; a DB-less run produces only `document.md`. Needs the three API keys
  below and the `mistral` extra (a plain `uv run` drops it — see HANDOFF gotchas).

## Conventions

- Keys (in `.env` — see `.env.example` — or environment secrets):
  `MISTRAL_API_KEY` (page OCR), `OPENROUTER_API_KEY` (correction pass, Qwen3-VL),
  `DEEPSEEK_API_KEY` (text stages).
- The package imports as `kms.*` (pyproject `package = false`); set `PYTHONPATH=src`. Internal
  imports are absolute (`from kms.core.state import ...`); dependencies point backward only
  (`core ← ingestion ← entity ← graph ← output`), never forward.
- Match the surrounding code's style. Parallel (map-reduce) stages use the
  `dispatch → worker → collect` shape; a genuinely sequential stage (e.g. `problem_finder`)
  is a plain graph node instead of forcing a single-Send fan-out.
