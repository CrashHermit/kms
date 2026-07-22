"""
Compile the instruction distributor with DSPy: bootstrap few-shot demonstrations for the
flash student, judged by the strong teacher (deepseek-v4-pro).

Unlike the splitter (page-sized windows → few, huge examples), the distributor's examples
are small — one lead-in plus a handful of following problems — so demo bootstrapping fits
well, and every stress-test run on disk contributes many lead-ins for free.

Loop:
  1. Reconstruct examples from finished run dirs (out/<fixture>/), or load live traces.
  2. Split train / dev.
  3. BootstrapFewShot runs the TEACHER over the train examples; the LLM judge
     (`distributor_score`) keeps only approved governance decisions as demos for the STUDENT.
  4. Report judge pass-rate: bare student vs compiled, on dev.
  5. Save to ``training/distributor/compiled.json`` (loaded by the serving node if present).

Run:  PYTHONPATH=src uv run python -m training.distributor.compile out/ea2e_ch1_review out/ea2e_sec1_3 out/lebl_ra1 ...
(args are run dirs; pass a single *.jsonl to use live traces instead.)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import dspy

from module.instruction_distributor import Module as DistributorModule
from module.llm import text_lm, teacher_lm
from training.distributor.dataset import load_runs, load_traces
from training.distributor.metric import distributor_score

OUT_DEFAULT = Path("training/distributor/compiled.json")


def _passrate(program, devset) -> float:
    ok = 0
    for ex in devset:
        pred = program(**ex.inputs())
        ok += 1 if distributor_score(ex, pred) else 0
    return ok / len(devset) if devset else 0.0


def _load(args: list[str]) -> list[dspy.Example]:
    if len(args) == 1 and args[0].endswith(".jsonl"):
        return load_traces(args[0])
    return load_runs(args)


def main(args: list[str], out_path: str | Path = OUT_DEFAULT) -> None:
    dspy.configure(lm=text_lm())  # student is the flash model
    examples = _load(args)
    if len(examples) < 4:
        raise SystemExit(f"need at least ~4 lead-in examples; got {len(examples)}")

    random.Random(0).shuffle(examples)
    cut = max(2, len(examples) // 4)
    devset, trainset = examples[:cut], examples[cut:]
    print(f"lead-in examples: {len(examples)}  train: {len(trainset)}  dev: {len(devset)}")

    student = DistributorModule(lm=text_lm(), compiled=False)  # start from the bare student
    base = _passrate(student, devset)
    print(f"baseline (bare student) dev judge pass-rate: {base:.2f}")

    optimizer = dspy.BootstrapFewShot(
        metric=distributor_score,
        teacher_settings={"lm": teacher_lm()},
        max_bootstrapped_demos=4,
        max_labeled_demos=0,  # reference-free: no gold outputs
        max_rounds=1,
    )
    compiled = optimizer.compile(student, trainset=trainset)

    after = _passrate(compiled, devset)
    n_demos = sum(len(getattr(p, "demos", [])) for _, p in compiled.named_predictors())
    print(f"compiled dev judge pass-rate: {after:.2f}  (baseline {base:.2f})  demos: {n_demos}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out_path))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: compile.py <run_dir> [<run_dir> ...] | <traces.jsonl>")
    main(argv, OUT_DEFAULT)
