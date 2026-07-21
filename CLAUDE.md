# KMS

Pipeline that turns a math textbook PDF into a knowledge graph of math entities
(Definitions, Theorems, Problems), following AutoMathKG (arXiv:2505.13406).

**Read `docs/HANDOFF.md` first** — it has the architecture, design decisions and their
rationale, validation results, full run instructions, and the container gotchas. This file
is just the quick reference.

## Current focus

The extraction front-end is **Mistral OCR + a vision correction pass** (no GPU), validated
on adversarial pages. The entity layer is mid-**redesign** into per-type finders over a
purely structural node stream: the **Problem finder is built and wired in**; the Definition
and Theorem finders, and the per-attribute passes (member roles, number, instruction, …),
are **not started** (see HANDOFF "entity-layer redesign"). The **graph tier** —
relationship/edge discovery, MathVD fusion, Math-LLM completion — is the next big piece
after that and is **not started**.

## Layout

- `src/module/` — the pipeline, wired by `pipeline.py` (LangGraph map-reduce; every stage is
  `dispatch → worker → collect`):
  `mistral_ocr → corrector → extractor → seam_merger → problem_finder`, then `assembler`
  runs after the graph. Two phases split at `seam_merger`: per-page ingestion (backbone
  `segments`) → flat global node stream (backbone `nodes`, stable ids). The extractor is
  **purely structural**; math-semantic typing lives entirely in the per-type entity finders
  (only `problem_finder` exists so far).
- `docs/HANDOFF.md` — full context.

## Commands

- Deps: `uv sync` (light CPU core) · `uv sync --extra mistral` (adds `pypdfium2` + `pillow`,
  used to render page images for the correction pass). **No GPU anywhere.**
- Tests: `PYTHONPATH=src uv run pytest -q` — `conftest` stubs the heavy deps, so it runs anywhere.
- Run: `PYTHONPATH=src uv run python -m module.pipeline book.pdf out/`, or from Python
  `run(pdf, output_dir="out/", pages=[...])` to limit pages. Writes `out/document.md` +
  `out/entities.json`. Needs the three API keys below.

## Conventions

- Keys (in `.env` — see `.env.example` — or environment secrets):
  `MISTRAL_API_KEY` (page OCR), `OPENROUTER_API_KEY` (correction pass, Qwen3-VL),
  `DEEPSEEK_API_KEY` (text stages).
- The package imports as `module.*` (pyproject `package = false`); set `PYTHONPATH=src`.
- Match the surrounding code's style; keep the `dispatch/worker/collect` shape for new stages.
