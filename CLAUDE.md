# KMS

Pipeline that turns a textbook PDF into a knowledge graph, following AutoMathKG
(arXiv:2505.13406) and AutoSchemaKG (arXiv:2505.23628). Math-first, but **generalized**: the domain
vocabularies are induced, not drawn from math taxonomies, so a physics or biology book ingests with
no new vocabulary.

**Read `docs/HANDOFF.md` first** — it has the architecture, design decisions and their
rationale, validation results, full run instructions, and the container gotchas. This file
is just the quick reference.

## Current focus

The extraction front-end is **Mistral OCR + a vision correction pass** (no GPU), validated on
adversarial pages. `docs/GENERALIZATION.md` is **built** (steps 1–4 of its build sequence); the two
things it leaves open are stated at the bottom of this section.

**The entity layer.** An exercise **splitter** makes exercises atomic at the node level and an
**instruction finder** tags exercise lead-in nodes `role="instruction"`. Then one of **two
interchangeable extraction layers** runs (`KMS_ENTITY_LAYER`, or `build_graph(layer)`):

- **`per-type`** (default, the validated math path): three **finders** (problem/definition/theorem)
  each build a sparse overlay, each followed by its **attributor**.
- **`block`** (the general path): one type-agnostic **block finder** emits spans only, a **universal
  attributor** induces each block's **open `type`** (definition / theorem / law / mechanism / …)
  alongside label/number/title/contents, and a **procedure finder** asks one direct question per
  entity — *is there something to work out, shown or absent?* — routing extract / create / defer /
  skip.

Both end at one **open-relation referencer** (target kind and relation both LLM-named, no closed
vocabulary — it replaced the three per-type referencers), with the **instruction distributor**
propagating a grouped-exercise lead-in's directive onto the tasks it governs. A **collector** then
flattens whichever overlays ran into one `entities` list; the **conceptualizer** tags every entity
and every procedure step with induced concepts (this replaced AutoMathKG's fixed `field`); the
**dependency finder** rolls those up along the reference graph into concept-level `:DEPENDS_ON`
prerequisites (reference-grounded, pairwise-judged, cycle-guarded into a DAG).

**The graph tier** (Neo4j) is the pipeline's persistence layer and the AutoMathKG + AutoSchemaKG
unified substrate (`docs/UNIFIED-KG.md`). **Seven built, wired layers**: the structural
**provenance** layer (a `:Source` per book rooting its `:Node` markdown stream via `:HEAD`/`:NEXT`);
the **`:Entity:Mention` overlay** on top of it (rooted via `:HAS_ENTITY`, linked to its member
`:Node` chunks via `:DERIVED_FROM`); the **procedural layer** (each derivation reified into a
`:Procedure` via `:HAS_PROCEDURE`, rooting an `:Event` step chain `:FIRST`/`:THEN`); the **concept
layer** (a global, born-canonical `:Concept` per induced tag, `:INSTANCE_OF` from entities *and*
procedure steps); the **prerequisite layer** (`(:Concept)-[:DEPENDS_ON {support}]->(:Concept)`, both
ends MATCHed, never minted); the **reference layer** (`:REFERENCES {relation}` edges onto a global
`:Entity:Canonical` **hub** per referenced target, so citations from any entity/book converge on one
node, plus step-level `:USES {relation}` edges from an `:Event` onto those same canonicals); and
`:REALIZES` edges tying each canonical **back** to the in-corpus mention that realizes it (nominal
title-match) — so a citation resolves *through* the shared hub to where the concept is actually
defined, the cross-corpus convergence payoff.

**One rule across every semantic layer: kind is the label, type is a property.** An induced type
cannot be a Neo4j label without the label set growing unbounded, so `:Entity`, `:Procedure`,
`:Concept` and the canonicals carry indexed `type` properties instead of `:Entity:Theorem`-style
per-type labels. All layers are wired into the entity persister and covered by the opt-in
integration test (`KMS_NEO4J_IT`). The graph **owns persistence** (the old `entities.json` /
`nodes.json` artifacts are gone).

**Left open, deliberately:** (1) the block layer has **not** been measured against the per-type
layer on real math books — the probes validated *concepts*, not general *extraction* — so the
per-type chains stay the default and are not deleted until parity is measured
(`docs/ENTITY-LAYER-REBUILD.md`); (2) the `:DEMONSTRATES`/`:PRACTICES` anchors and the **semantic**
dedup tier (embedding fusion, which refines `:REALIZES` and merges concept paraphrases) are not
started.

## Layout

- `src/kms/` — the pipeline, organized by phase (see `docs/ARCHITECTURE.md` for the full
  rationale and the backward-only dependency rule). Packages:
  - `core/` — shared center that every stage depends on and that depends on no stage:
    `models.py` (domain data, dspy/langgraph-free), `state.py` (the LangGraph `State`),
    `llm.py` (LM config).
  - `ingestion/` — phase 1 (backbone `segments`): `ocr.py` (Mistral front-end), `corrector.py`,
    `extractor.py` (purely structural), `seam_merger.py`. Map-reduce `dispatch → worker → collect`.
  - `entity/` — phase 2 (backbone `nodes`): `splitter.py`, `instruction_finder.py`,
    `instruction_distributor.py`, `referencers/open.py`, `collector.py`, `conceptualizer.py`,
    `dependency_finder.py`; plus the two extraction layers — `finders/block.py` +
    `attributors/universal.py` + `procedure_finder.py` (general), and
    `finders/{problem,definition,theorem}.py` + `attributors/{problem,definition,theorem}.py`
    (the validated math path). Plain sequential nodes.
  - `output/` — `assembler.py` (runs after the graph).
  - `graph/` — phase 3 (Neo4j). All seven layers built (`docs/UNIFIED-KG.md`): `db.py` (async
    driver, the only neo4j import; plus an `NEO4J_TRANSPORT=http` HTTPS Query-API transport for
    sandboxes where Bolt/7687 is blocked), `nodes.py` (ASTNode→Neo4j), `entities.py` (Entity→Neo4j,
    `:Entity:Mention`, type as a property), `procedures.py` (`procedures`→`:Procedure`/`:Event`),
    `concepts.py` (induced tags→global `:Concept` + `:INSTANCE_OF`, entities and steps),
    `dependencies.py` (concept `:DEPENDS_ON`), `references.py` (refs→`:Entity:Canonical` hubs +
    `:REFERENCES` edges), `uses.py` (step-level `:Event`→`:Canonical` `:USES` edges) and
    `realizes.py` (`:Mention`→`:Canonical` `:REALIZES` identity edges, nominal title-match) — all
    deterministic uuids, driver-free; `schema.py` (constraint/index bootstrap for all layers),
    `writer.py` (one `persist_*` per layer), `persister.py` (the two pipeline stages:
    `NodePersisterNode`, `EntityPersisterNode` — the latter persists every layer above the nodes, in
    dependency order). Not started: the `:DEMONSTRATES`/`:PRACTICES` anchors and the semantic dedup
    that refines `:REALIZES` (embedding fusion).
  - `pipeline.py` wires the graph; `cli.py` is the `__main__` entry; `kms/__init__.py` exposes `run`.
- Flow: `ocr → corrector → extractor → seam_merger → splitter → instruction_finder →
  node_persister → <entity layer> → collector → conceptualizer → dependency_finder →
  entity_persister`. Two phases split at `seam_merger`: per-page ingestion (backbone `segments`) →
  flat global node stream (backbone `nodes`, stable ids). The **splitter** rewrites `nodes` so each
  exercise (and each embedded lead-in) is its own node; the **instruction finder** then tags every
  lead-in node `role="instruction"` over that atomic stream. The `node_persister` stage (after the
  splitter, before the finders) writes the node stream to Neo4j as the `:Source`/`:Node` provenance
  layer. The **entity layer** is the one swappable span — three parallel per-type chains, or the one
  block chain — and the **collector** is the seam that makes them interchangeable: it flattens
  whichever overlays ran into one document-ordered, globally-id'd `entities` list, so everything
  downstream is identical either way. The `entity_persister` then writes the `:Entity` overlay and,
  on top of it, the procedural, concept, `:DEPENDS_ON`, reference, step-level `:USES` and
  `:REALIZES` layers. All persistence is gated on Neo4j being configured (`NEO4J_*` env vars) and is
  a no-op otherwise, so a DB-less run still produces `document.md` but persists nothing.
- `docs/HANDOFF.md` — full context. `docs/ARCHITECTURE.md` — the package layout and its rules.
  `docs/GENERALIZATION.md` — the generalization design. `docs/ENTITY-LAYER-REBUILD.md` — what to
  remove once the block layer reaches parity.

## Commands

- Deps: `uv sync` (light CPU core) · `uv sync --extra mistral` (adds `pypdfium2` + `pillow`,
  used to render page images for the correction pass). **No GPU anywhere.**
- Tests: `PYTHONPATH=src uv run pytest -q` (193 tests) — `conftest` stubs the heavy deps, so it
  runs anywhere, no keys needed.
- Run (full pipeline): `PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/`,
  or from Python `from kms import run; run(pdf, output_dir="out/", pages=[...])` to limit pages
  (0-based). Writes `out/document.md` and, when Neo4j is configured (`NEO4J_*`), persists the graph;
  a DB-less run produces only `document.md`. Needs the three API keys below and the `mistral` extra
  (a plain `uv run` drops it — see HANDOFF gotchas).
- Entity layer: `KMS_ENTITY_LAYER=block` runs the general path instead of the validated per-type
  one — that is how the two are compared on the same book.

## Conventions

- Keys (in `.env` — see `.env.example` — or environment secrets):
  `MISTRAL_API_KEY` (page OCR), `OPENROUTER_API_KEY` (correction pass, Qwen3-VL),
  `DEEPSEEK_API_KEY` (text stages).
- The package imports as `kms.*` (pyproject `package = false`); set `PYTHONPATH=src`. Internal
  imports are absolute (`from kms.core.state import ...`); dependencies point backward only
  (`core ← ingestion ← entity ← graph ← output`), never forward.
- Match the surrounding code's style. Parallel (map-reduce) stages use the
  `dispatch → worker → collect` shape; a genuinely sequential stage (e.g. `block_finder`)
  is a plain graph node instead of forcing a single-Send fan-out.
- **Kind is the label, type is a property** in the graph tier — an induced vocabulary never becomes
  a Neo4j label set. Closed vocabularies (`NodeType`, `ACTIONS_ALL`) stay closed and keep their
  labels; open ones (entity/procedure/concept types, relations) ride as indexed properties.
