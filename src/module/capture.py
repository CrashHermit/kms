"""Optional trace capture for building DSPy trainsets.

When the ``KMS_CAPTURE_DIR`` environment variable is set, instrumented stages append
one JSON line per call to ``<KMS_CAPTURE_DIR>/<signature>.jsonl``, recording the exact
signature inputs and the produced outputs. A normal run leaves this off and pays
nothing. Curate the resulting JSONL into trainsets and load them with ``trainsets.py``.

Each line is ``{"inputs": {...}, "outputs": {...}}`` where the keys are the signature's
input/output field names, so a line maps directly onto a ``dspy.Example``.
"""

import json
import os
from pathlib import Path


def enabled() -> bool:
    return bool(os.environ.get("KMS_CAPTURE_DIR"))


def record(signature: str, inputs: dict, outputs: dict) -> None:
    """Append one ``{inputs, outputs}`` example for ``signature``. No-op when capture
    is disabled, and never raises into the pipeline — tracing must not break a run."""
    directory = os.environ.get("KMS_CAPTURE_DIR")
    if not directory:
        return
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"inputs": inputs, "outputs": outputs}, ensure_ascii=False)
        with (path / f"{signature}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass
