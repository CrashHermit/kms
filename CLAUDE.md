# KMS

Pipeline that turns a math textbook PDF into a knowledge graph of math entities
(Definitions, Theorems, Problems), following AutoMathKG (arXiv:2505.13406).

**Read `docs/HANDOFF.md` first** — it has the architecture, design decisions and their
rationale, validation results, full run instructions, and the container gotchas. This file
is just the quick reference.

## Current focus

The extraction front-end is **Mistral OCR + a vision correction pass** (no GPU), validated
on adversarial pages. The entity layer is **built**: an exercise **splitter** makes exercises
atomic at the node level, three per-type **finders** (problem/definition/theorem) each build a
sparse overlay, three per-type **attributors** fill the self-contained AutoMathKG attributes,
and an **instruction distributor** propagates a grouped-exercise lead-in's directive onto the
Problems it governs. The **graph tier** (Neo4j) is now **started**: its structural provenance
layer — a `:Source` node per book rooting its `:Node` markdown stream via `:HEAD`/`:NEXT` edges
(reusing `core.NodeType`) — is built, wired into the pipeline after the splitter, and validated
end-to-end against a real Neo4j. Still to come: the semantic tiers (dedup canonicals, general
entities, concepts), cross-entity `refs`/`references_tactics`, MathVD fusion, and Math-LLM
completion.

## Layout

- `src/kms/` — the pipeline, organized by phase (see `docs/ARCHITECTURE.md` for the full
  rationale and the backward-only dependency rule). Packages:
  - `core/` — shared center that every stage depends on and that depends on no stage:
    `models.py` (domain data, dspy/langgraph-free), `state.py` (the LangGraph `State`),
    `llm.py` (LM config), `tracing.py`.
  - `ingestion/` — phase 1 (backbone `segments`): `ocr.py` (Mistral front-end), `corrector.py`,
    `extractor.py` (purely structural), `seam_merger.py`. Map-reduce `dispatch → worker → collect`.
  - `entity/` — phase 2 (backbone `nodes`): `splitter.py`, `finders/{problem,definition,theorem}.py`,
    `attributors/{problem,definition,theorem}.py`, and `instruction_distributor.py` (problem chain only).
    Plain sequential nodes.
  - `output/` — `assembler.py` (runs after the graph).
  - `graph/` — phase 3 (Neo4j). **Structural provenance layer built**: `db.py` (async driver, the
    only neo4j import), `nodes.py` (ASTNode→Neo4j mapping, deterministic uuids, multi-label),
    `schema.py` (constraint/index bootstrap), `writer.py` (`persist_nodes`), `persister.py` (the
    pipeline stage). The semantic tiers (dedup canonicals, general entities, concepts, cross-entity
    refs/tactics, MathVD fusion, Math-LLM completion) are **not started**.
  - `pipeline.py` wires the graph; `cli.py` is the `__main__` entry; `kms/__init__.py` exposes `run`.
- Flow: `ocr → corrector → extractor → seam_merger → splitter → node_persister →
  {problem,definition,theorem} finder → {…} attributor`, and the problem chain has one more stage,
  the instruction distributor. Two phases split at `seam_merger`: per-page ingestion (backbone
  `segments`) → flat global node stream (backbone `nodes`, stable ids). The **splitter** rewrites
  `nodes` so each exercise is its own node and tags lead-ins `role="instruction"`. The three finders
  walk `nodes` in parallel and write their own entity channel; each attributor enriches its channel
  in place; `run()` concatenates the three into one flat, document-ordered `entities.json`
  (`[{id, type, members, …attrs}]`, overlap is fine — members are node-id pointers) and persists the
  node stream to `nodes.json` for provenance. The `node_persister` stage (after the splitter, before
  the finders) also writes that stream to Neo4j as the `:Source`/`:Node` provenance layer when Neo4j
  is configured (`NEO4J_*` env vars), and is a no-op otherwise so DB-less runs are unchanged. The
  finders (and attributors) are self-contained copies
  of one shape.
- `docs/HANDOFF.md` — full context. `docs/ARCHITECTURE.md` — the package layout and its rules.

## Commands

- Deps: `uv sync` (light CPU core) · `uv sync --extra mistral` (adds `pypdfium2` + `pillow`,
  used to render page images for the correction pass). **No GPU anywhere.**
- Tests: `PYTHONPATH=src uv run pytest -q` (46 tests) — `conftest` stubs the heavy deps, so it
  runs anywhere, no keys needed.
- Run (full pipeline): `PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/`,
  or from Python `from kms import run; run(pdf, output_dir="out/", pages=[...])` to limit pages
  (0-based). Writes
  `out/document.md` + `out/entities.json` + `out/nodes.json`. Needs the three API keys below and
  the `mistral` extra (a plain `uv run` drops it — see HANDOFF gotchas).

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
