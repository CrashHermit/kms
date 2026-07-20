#!/bin/bash
# Prepare a Claude Code on the web session: install the light LLM-pipeline deps so the
# tests (and the pipeline's LLM stages) run out of the box. The heavy docling/torch
# extraction extra is deliberately skipped — web has no GPU; install it locally with
# `uv sync --extra extract`.
set -euo pipefail

# Web sessions only; local dev manages its own environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
uv sync

# Persist into the session so `python`/`pytest` use the project venv and `module`
# (which lives under src/, with pyproject package=false) is importable.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
    echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR/src\""
  } >> "$CLAUDE_ENV_FILE"
fi
