"""Compile few-shot demos for the LLM signatures via DSPy teacher-student bootstrapping.

A stronger *teacher* model (``llm.teacher_lm``, set via ``TEACHER_MODEL``) generates
demonstration traces on the captured trainset; a per-signature *metric* keeps only the
correct ones; the survivors are compiled as few-shot demos for the production *student*
model (``llm.text_lm``). The cheap fast model then runs in production carrying demos a
more capable model produced once.

    TEACHER_MODEL=<stronger-deepseek> PYTHONPATH=src \
        uv run python -m module.optimize entity_attributor entity_grouper

Compiled programs are written to ``data/compiled/<signature>.json`` and can be loaded
into the matching stage's predictor with ``load_into``.
"""

import sys
from pathlib import Path

import dspy

from . import trainsets
from .llm import text_lm, teacher_lm
from .entity_attributor import Signature as AttributorSignature
from .entity_grouper import Signature as GrouperSignature

COMPILED_DIR = Path(__file__).resolve().parents[2] / "data" / "compiled"


# --- metrics: exact-match on the structured output, per signature ---

def _as_str_list(value) -> list[str]:
    return [str(v) for v in (value or [])]


def attributor_metric(example, pred, trace=None) -> bool:
    """The predicted role sequence exactly matches the gold roles."""
    return _as_str_list(getattr(pred, "roles", None)) == _as_str_list(getattr(example, "roles", None))


def _span_set(obj) -> set:
    return {(str(s.type), int(s.start), int(s.end)) for s in (getattr(obj, "entities", None) or [])}


def grouper_metric(example, pred, trace=None) -> bool:
    """The predicted set of (type, start, end) spans exactly matches the gold spans."""
    return _span_set(pred) == _span_set(example)


SIGNATURES = {
    "entity_attributor": (AttributorSignature, attributor_metric),
    "entity_grouper": (GrouperSignature, grouper_metric),
}


def compile_signature(name: str, max_demos: int = 4):
    """Bootstrap-compile few-shot demos for one signature and save the program."""
    signature, metric = SIGNATURES[name]
    trainset = trainsets.load(name)
    if not trainset:
        raise SystemExit(f"no trainset for {name} (data/trainsets/{name}.jsonl is empty/missing)")

    student = dspy.ChainOfThought(signature)
    student.set_lm(text_lm())

    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=min(len(trainset), 16),
        teacher_settings={"lm": teacher_lm()},
    )
    compiled = optimizer.compile(student, trainset=trainset)

    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    out = COMPILED_DIR / f"{name}.json"
    compiled.save(str(out))

    n_demos = sum(len(getattr(p, "demos", []) or []) for p in compiled.predictors())
    print(f"{name}: {len(trainset)} trainset examples -> {n_demos} demos -> {out}")
    return compiled


def load_into(predictor: dspy.Module, name: str) -> bool:
    """Load a previously compiled program's demos into `predictor`, if one exists.
    Returns True if loaded. Use in a stage Module to pick up optimized demos when
    available and fall back to the base prompt otherwise."""
    path = COMPILED_DIR / f"{name}.json"
    if not path.exists():
        return False
    predictor.load(str(path))
    return True


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(SIGNATURES)):
        compile_signature(name)
