#!/bin/bash
# Prepare a Claude Code on the web session: install the light LLM-pipeline deps
# so the tests (and the pipeline's LLM stages) run out of the box. The optional
# `mistral` extra (pypdfium2/pillow, for page rendering) is left out — install
# it with `uv sync --extra mistral` when a run needs the OCR front-end.
set -euo pipefail

# Web sessions only; local dev manages its own environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
uv sync

# Persist into the session so `python`/`pytest` use the project venv and `kms`
# (which lives under src/) is importable.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
    echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR/src\""
  } >> "$CLAUDE_ENV_FILE"
fi
