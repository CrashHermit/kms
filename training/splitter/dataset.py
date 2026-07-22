"""
Build a splitter trainset from captured traces.

Because the metric is a reference-free LLM judge, a training example needs only the
INPUT the splitter saw — the window of nodes. We read the ``splitter.jsonl`` traces
(written when a run sets ``KMS_TRACE_DIR``), rebuild each window as ``WindowNode``
objects, de-duplicate identical windows, and wrap them as ``dspy.Example``s with
``current_nodes`` marked as the input field. The optimizer runs the student over these
and keeps the outputs the judge approves.
"""

import json
from pathlib import Path

import dspy

from module.exercise_splitter import WindowNode


def load_windows(traces_path: str | Path) -> list[dspy.Example]:
    """Load unique splitter input windows from a ``splitter.jsonl`` trace file."""
    path = Path(traces_path)
    seen: set[str] = set()
    examples: list[dspy.Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        nodes = rec.get("inputs", {}).get("current_nodes") or []
        key = json.dumps(nodes, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        window = [WindowNode(**n) for n in nodes]
        examples.append(dspy.Example(current_nodes=window).with_inputs("current_nodes"))
    return examples
