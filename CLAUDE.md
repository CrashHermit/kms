# KMS

Pipeline that turns a textbook PDF into a knowledge graph of **pedagogical blocks**, the
**procedures** that derive them, and their **steps**.

**Read `docs/HANDOFF.md` first** — status, validation history, run instructions, container
gotchas. Then `docs/SCHEMA.md` (what the graph is) and `docs/REBUILD.md` (how it got here).
This file is the quick reference.

## Current focus

The extraction front-end is **Mistral OCR + a vision correction pass** (no GPU), validated on
adversarial pages. The **entity layer was rebuilt** to be domain-general: the AutoMathKG
structures are gone (see `docs/REBUILD.md`), and nine per-type modules collapsed into one chain —
**group finder → role typer → block typer → statement extractor → procedure extractor**, each
stage asking exactly ONE question.

- The **group finder** walks the node stream once and cuts it into **untyped** spans — boundaries
  only. A theorem and its proof are **two adjacent spans**, so the statement/derivation boundary
  is a structural detection, not a semantic call; the cut lands where the text stops posing or
  asserting and starts working, marked (`Proof.`) or not. One partition — a node belongs to at
  most one span.
- The **role typer** makes one closed, binary call per span: `entity` (a block) or `procedure`
  (the derivation that resolves one). A span that opens with its own label is a block whatever
  follows it; an unlabelled computation session is the working.
- The **block typer** induces the **open `type`** (definition / theorem / example / law /
  mechanism / …) — a property on a bare `:Entity`, never a Neo4j label. It types the block, never
  its subject matter.
- The **statement extractor** transcribes the rest: label, number, title, contents.
- The **procedure extractor** decomposes every procedure span into **verbatim** ordered steps and
  attaches it to the nearest preceding block. Decomposition is **universal**: solutions get steps
  too (AutoMathKG restricted its step list to Thm/Def, leaving every solution stepless).

The **graph tier** (Neo4j) is the persistence layer. Four semantic kinds — `:Entity`,
`:Procedure`, `:Act`, `:Concept` — over the `:Source`/`:Node` provenance tier. The **concept layer
is currently dark**: its only source was the deleted `field` taxonomy, so nothing writes
`:Concept` nodes or `:INSTANCE_OF` edges yet. Since block-to-block relations are also gone,
**blocks currently connect to nothing outside their own `:Source`** — building the concept layer
(`docs/CONCEPT-LAYER.md`) is the highest-value next step, after re-running the fixture books.

## Layout

- `src/kms/` — the pipeline, organized by phase (see `docs/ARCHITECTURE.md` for the rationale and
  the backward-only dependency rule). Packages:
  - `core/` — shared center that every stage depends on and that depends on no stage:
    `models.py` (domain data, dspy/langgraph-free), `state.py` (the LangGraph `State`),
    `llm.py` (LM config), `logs.py` (log formatting helpers). There is no `tracing.py` —
    trainable per-call capture was removed with the AutoMathKG layer and has not been
    replaced; `logs.py` is a debugging aid, not a substitute (see `docs/HANDOFF.md`).
  - `ingestion/` — phase 1 (backbone `segments`): `ocr.py` (Mistral front-end), `corrector.py`,
    `extractor.py` (purely structural), `seam_merger.py`. Map-reduce `dispatch → worker → collect`.
  - `entity/` — phase 2 (backbone `nodes`), all plain sequential nodes: `splitter.py`,
    `instruction_finder.py`, `group_finder.py`, `role_typer.py`, `block_typer.py`,
    `statement_extractor.py`, `procedure_extractor.py`, `instruction_distributor.py`.
    One question per stage: the finder cuts boundaries, `role_typer` says block-or-derivation
    (closed, binary), `block_typer` induces the open `type`, and the statement extractor
    transcribes label/number/title/contents.
  - `graph/` — phase 3 (Neo4j): `db.py` (async driver, the only neo4j import; plus an
    `NEO4J_TRANSPORT=http` HTTPS Query-API transport for sandboxes where Bolt/7687 is blocked),
    `nodes.py` (ASTNode→`:Node`), `entities.py` (Entity→bare `:Entity`),
    `procedures.py` (Procedure→`:Procedure` + its `:Act` chain), `concepts.py` (hub identity only
    — **dark**), `schema.py` (constraint/index bootstrap), `writer.py` (`persist_nodes` +
    `persist_entities` + `persist_procedures`), `persister.py` (`NodePersisterNode`,
    `EntityPersisterNode`). All mapping is deterministic-uuid and driver-free.
  - `output/` — `assembler.py` (runs after the graph).
- `pipeline.py` wires the graph; `cli.py` is the `__main__` entry; `kms/__init__.py` exposes `run`.
- Flow: `ocr → corrector → extractor → seam_merger → splitter → instruction_finder →
  node_persister → group_finder → role_typer → block_typer → statement_extractor →
  procedure_extractor → instruction_distributor → entity_persister`. Two phases split at `seam_merger`: per-page
  ingestion (backbone `segments`) → flat global node stream (backbone `nodes`, stable ids). The
  splitter makes each exercise its own node; the instruction finder then tags lead-ins
  `role="instruction"` over that atomic stream. `node_persister` writes the stream as the
  `:Source`/`:Node` provenance layer; `entity_persister` orders the overlay and writes the
  `:Entity` and `:Procedure`/`:Act` layers. Both persist only when Neo4j is configured
  (`NEO4J_*`) and are no-ops otherwise, so a DB-less run still produces `document.md`.

## Commands

- Deps: `uv sync` (light CPU core) · `uv sync --extra mistral` (adds `pypdfium2` + `pillow`,
  used to render page images for the correction pass). **No GPU anywhere.**
- Logs: every stage logs one INFO line summarising what it produced; `KMS_LOG_LEVEL=DEBUG` adds
  one line per DSPy call (inputs' shape + elided outputs, ~70 lines/book). Loggers are named after
  their modules, so a single stage can be turned up on its own.
- Tests: `PYTHONPATH=src uv run pytest -q` (161 tests, 3 skipped) — `conftest` stubs the heavy
  deps, so it runs anywhere, no keys needed. The Neo4j integration test is opt-in
  (`KMS_NEO4J_IT=1`).
- Run (full pipeline): `PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/`,
  or from Python `from kms import run; run(pdf, output_dir="out/", pages=[...])` to limit pages
  (0-based). Writes `out/document.md` and, when Neo4j is configured, persists the graph. Needs the
  three API keys below and the `mistral` extra (a plain `uv run` drops it — see HANDOFF gotchas).

## Conventions

- Keys (in `.env` — see `.env.example` — or environment secrets):
  `MISTRAL_API_KEY` (page OCR), `OPENROUTER_API_KEY` (correction pass, Qwen3-VL),
  `DEEPSEEK_API_KEY` (text stages).
- The package imports as `kms.*` (pyproject `package = false`); set `PYTHONPATH=src`. Internal
  imports are absolute (`from kms.core import models`); dependencies point backward only
  (`core ← ingestion ← entity ← graph ← output`), never forward.
- **Persisted vs. transient** (`docs/SCHEMA.md`, principle 1): the in-memory models are the
  pipeline's working state, the graph is the deliverable. A field is persisted only if something
  reads it *from the graph*. `ASTNode.role` and all of `Segment` are transient.
- **Kind is a label, type is a property.** A node kind is a Neo4j label; a subtype is a property,
  and only exists if it is non-constant and not derivable from a neighbour. Open type sets never
  become labels.
- Match the surrounding code's style (`docs/STYLE.md` — Google Python, 80 cols, single quotes,
  module-level imports). Parallel (map-reduce) stages use the `dispatch → worker → collect` shape;
  a genuinely sequential stage (every entity stage) is a plain graph node instead of forcing a
  single-Send fan-out.
