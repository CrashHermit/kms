"""
Build a distributor trainset — one example per tagged lead-in.

``load_traces`` reads a ``distributor.jsonl`` captured live via ``KMS_TRACE_DIR`` on a pipeline
run and yields ``dspy.Example(lead_in, following_problems)`` inputs (the metric is a reference-free
judge, so no gold is needed). It feeds the compile step directly.

(An earlier ``load_runs`` path reconstructed the same examples from run artifacts on disk
(``nodes.json`` + ``entities.json``); those files were retired when the graph became the pipeline's
store, so live trace capture is now the single source.)
"""

import json
from pathlib import Path

import dspy

from kms.entity.instruction_distributor import WindowProblem


def load_traces(traces_path: str | Path) -> list[dspy.Example]:
    """Examples from a live-captured ``distributor.jsonl`` (KMS_TRACE_DIR)."""
    path = Path(traces_path)
    seen: set[str] = set()
    out: list[dspy.Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ins = rec.get("inputs", {})
        lead_in = ins.get("lead_in") or ""
        fp = ins.get("following_problems") or []
        key = json.dumps({"lead_in": lead_in, "fp": fp}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        following = [WindowProblem(**p) for p in fp]
        out.append(
            dspy.Example(lead_in=lead_in, following_problems=following).with_inputs(
                "lead_in", "following_problems"
            )
        )
    return out
