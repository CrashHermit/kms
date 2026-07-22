"""
Build a distributor trainset — one example per tagged lead-in.

Two sources, both yielding ``dspy.Example(lead_in, following_problems)`` inputs (the
metric is a reference-free judge, so no gold is needed):

  * ``load_runs`` reconstructs examples straight from finished run artifacts
    (``out/<fixture>/nodes.json`` + ``entities.json``) — it replays the distributor's
    positional bracketing (a lead-in's candidates are the problems between it and the
    next lead-in) with zero new pipeline calls. This is the cheap way to seed from every
    stress-test run we already have on disk.
  * ``load_traces`` reads ``distributor.jsonl`` captured live via ``KMS_TRACE_DIR``.

Both feed the same compile step.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from module.instruction_distributor import WindowProblem


def _problem_text(contents, number) -> str:
    body = " ".join(c for c in (contents or []) if c)
    return body or (number or "")


def _example(lead_in: str, candidates: list[dict]) -> dspy.Example:
    following = [
        WindowProblem(position=k, number=c.get("number"), text=_problem_text(c.get("contents"), c.get("number")))
        for k, c in enumerate(candidates)
    ]
    return dspy.Example(lead_in=lead_in, following_problems=following).with_inputs("lead_in", "following_problems")


def reconstruct_from_run(run_dir: str | Path) -> list[dspy.Example]:
    """One example per lead-in in a finished run dir, via positional bracketing."""
    run = Path(run_dir)
    nodes = json.loads((run / "nodes.json").read_text(encoding="utf-8"))
    ents = json.loads((run / "entities.json").read_text(encoding="utf-8"))
    # Node id == stream position (ids are assigned 0..n-1 in document order after the splitter).
    lead_ids = sorted(n["id"] for n in nodes if n.get("role") == "instruction")
    node_content = {n["id"]: n.get("content") or "" for n in nodes}
    problems = sorted(
        (e for e in ents if e["type"] == "problem" and e.get("members")),
        key=lambda e: e["members"][0],
    )
    far = len(nodes)
    out: list[dspy.Example] = []
    for i, here in enumerate(lead_ids):
        nxt = lead_ids[i + 1] if i + 1 < len(lead_ids) else far
        candidates = [e for e in problems if here < e["members"][0] < nxt]
        if candidates:
            out.append(_example(node_content.get(here, ""), candidates))
    return out


def load_runs(run_dirs: list[str | Path]) -> list[dspy.Example]:
    """Reconstructed examples across several run dirs, de-duplicated by input."""
    seen: set[str] = set()
    out: list[dspy.Example] = []
    for d in run_dirs:
        for ex in reconstruct_from_run(d):
            key = json.dumps(
                {"lead_in": ex.lead_in, "fp": [p.model_dump() for p in ex.following_problems]},
                sort_keys=True, ensure_ascii=False,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(ex)
    return out


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
        out.append(dspy.Example(lead_in=lead_in, following_problems=following).with_inputs("lead_in", "following_problems"))
    return out
