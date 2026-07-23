"""
Opt-in trace capture for DSPy stages — the raw material for the data→compile loop.

When the ``KMS_TRACE_DIR`` environment variable is set, a stage may call
``record(stage, inputs, outputs, **meta)`` to append one JSON line to
``$KMS_TRACE_DIR/<stage>.jsonl`` for every LLM call it makes. Off by default the
call is a no-op, so instrumenting a stage costs nothing at inference time.

These traces are the honest, per-call I/O each stage actually saw. We curate them
into gold ``dspy.Example`` trainsets (accept the correct ones, correct the wrong
ones) and hand those to a DSPy optimizer, so stage quality is driven by data +
a metric instead of hand-tuned prompts. Traces are captured from the *actual*
input a stage received, which keeps upstream failures (e.g. an OCR-dropped
lead-in) from being mislabeled as this stage's error.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def trace_dir() -> Path | None:
    """The active trace directory, or None when capture is disabled."""
    d = os.environ.get("KMS_TRACE_DIR")
    return Path(d) if d else None


def record(stage: str, inputs: dict[str, Any], outputs: dict[str, Any], **meta: Any) -> None:
    """Append one ``{stage, inputs, outputs, **meta}`` line to ``<stage>.jsonl``.

    No-op unless ``KMS_TRACE_DIR`` is set. Best-effort: capture must never break a
    run, so any I/O error is swallowed. Thread-safe (stages fan out concurrently)."""
    d = trace_dir()
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"stage": stage, **meta, "inputs": inputs, "outputs": outputs}, ensure_ascii=False
        )
        with _lock, (d / f"{stage}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, TypeError, ValueError):
        # Best-effort telemetry: a filesystem error (OSError) or a non-serializable
        # payload (TypeError/ValueError from json.dumps) must never break a run.
        pass
