"""
Compile the exercise splitter with DSPy: bootstrap few-shot demonstrations for the
flash student, judged by the strong teacher (deepseek-v4-pro).

Loop:
  1. Load real splitter input windows captured from fixture runs (traces).
  2. Split into train / dev.
  3. BootstrapFewShot runs the TEACHER over the train windows, and the LLM judge
     (`splitter_score`, also on the teacher) keeps only the outputs it approves as
     demonstrations for the flash STUDENT.
  4. Evaluate judge pass-rate for the bare student vs the compiled student on dev.
  5. Save the compiled program to ``training/splitter/compiled.json`` — the serving
     ``SplitterNode`` loads it if present.

Run:  PYTHONPATH=src uv run python -m training.splitter.compile out/traces/splitter.jsonl
(A second arg overrides the output path.)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import dspy

from module.exercise_splitter import Module as SplitterModule
from module.llm import text_lm, teacher_lm
from training.splitter.dataset import load_windows
from training.splitter.metric import splitter_score

OUT_DEFAULT = Path("training/splitter/compiled.json")


def _passrate(program, devset) -> float:
    ok = 0
    for ex in devset:
        pred = program(**ex.inputs())
        ok += 1 if splitter_score(ex, pred) else 0
    return ok / len(devset) if devset else 0.0


def main(traces_path: str, out_path: str | Path = OUT_DEFAULT) -> None:
    dspy.configure(lm=text_lm())  # student is the flash model
    examples = load_windows(traces_path)
    if len(examples) < 4:
        raise SystemExit(f"need at least ~4 windows to compile; got {len(examples)} from {traces_path}")

    random.Random(0).shuffle(examples)
    cut = max(1, len(examples) // 5)
    devset, trainset = examples[:cut], examples[cut:]
    print(f"windows: {len(examples)}  train: {len(trainset)}  dev: {len(devset)}")

    student = SplitterModule(lm=text_lm(), compiled=False)  # start from the bare student
    base = _passrate(student, devset)
    print(f"baseline (bare student) dev judge pass-rate: {base:.2f}")

    optimizer = dspy.BootstrapFewShot(
        metric=splitter_score,
        teacher_settings={"lm": teacher_lm()},
        max_bootstrapped_demos=4,
        max_labeled_demos=0,  # reference-free: no gold outputs to use as labeled demos
        max_rounds=1,
    )
    compiled = optimizer.compile(student, trainset=trainset)

    after = _passrate(compiled, devset)
    print(f"compiled dev judge pass-rate: {after:.2f}  (baseline {base:.2f})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out_path))
    n_demos = len(getattr(compiled.splitter, "demos", []))
    print(f"saved {n_demos} demo(s) -> {out_path}")


if __name__ == "__main__":
    traces = sys.argv[1] if len(sys.argv) > 1 else "out/traces/splitter.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT
    main(traces, out)
