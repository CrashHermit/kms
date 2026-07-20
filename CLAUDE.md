# KMS

Pipeline that turns a math textbook PDF into a knowledge graph of math entities
(Definitions, Theorems, Problems), following AutoMathKG (arXiv:2505.13406).

**Read `docs/HANDOFF.md` first** — it has the architecture, design decisions and their
rationale, validation results, full run instructions, and the container gotchas. This file
is just the quick reference.

## Current focus: extraction

The entity layer (grouping + role attribution) is built and validated on two books. The
active priority is **hardening the extraction front-end** — `ocr.py`, `extractor.py`,
`seam_merger.py` — so downstream stages get clean, faithful nodes. The two known artifacts
(fused statement+proof; duplicated content on complex layouts) originate there. The graph
tier is **deferred** until extraction is solid (see HANDOFF "Next steps").

## Layout

- `src/module/` — pipeline stages, LangGraph map-reduce; every stage is
  `dispatch → worker → collect`. Two phases split at `seam_merger`: per-page ingestion
  (backbone `segments`) → flat global node stream (backbone `nodes`, stable ids).
- `docs/HANDOFF.md` — full context. `scripts/pdf_to_segments.py` — render a PDF to a test tree.
- `data/trainsets/` — captured DSPy examples (`capture.py` / `trainsets.py` / `optimize.py`).

## Commands

- Deps: `uv sync` (light CPU core) · `uv sync --extra extract` (docling + torch, needs a GPU).
- Tests: `PYTHONPATH=src uv run pytest -q` — `conftest` stubs the heavy deps, so it runs anywhere.
- Run without a GPU: render pages with `scripts/pdf_to_segments.py`, then
  `run(..., extract_pictures=False)` (docling bypass — see HANDOFF "how to run").

## Conventions

- Keys: `DEEPSEEK_API_KEY` (text stages), `OPENROUTER_API_KEY` (vision stages) — in `.env`
  (see `.env.example`) or environment secrets.
- The package imports as `module.*` (pyproject `package = false`); set `PYTHONPATH=src`.
- Match the surrounding code's style; keep the `dispatch/worker/collect` shape for new stages.
