"""
LLM-as-judge metric for the exercise-splitter DSPy program.

Instead of hand-scoring split boundaries and tag sets, we let a strong model grade
the flash splitter's output. The judge is reference-free: given the window of nodes
the splitter saw and what it produced, it decides whether the normalisation is
correct against the splitter's spec — packed exercise runs carved into one verbatim
piece per exercise, true lead-ins tagged, own-numbered exercises left untagged.

This keeps the trainset cheap (inputs only, no gold labels) and lets the optimizer
bootstrap: run the student over real windows, keep the outputs the judge approves as
few-shot demonstrations. The judge runs on the teacher model (deepseek-v4-pro) so the
grader is stronger than the student it grades.
"""

import dspy

from kms.core.llm import teacher_lm


class JudgeSplit(dspy.Signature):
    r"""
    You are grading a SPLITTER that normalises a window of textbook nodes for the
    exercise layer. It does two jobs and you judge whether BOTH are done correctly.

    1. SPLIT: any single node that packs TWO OR MORE numbered exercises must be carved
       into one piece per exercise — verbatim text, subparts (a)(b)(c) kept together,
       reference number captured, nothing paraphrased, dropped, or duplicated. A node
       with a single exercise, a worked example, prose, or a header must NOT be split.
    2. TAG (instruction_positions): a node that is an exercise LEAD-IN — a directive with
       NO number of its own that introduces a run of other, separately-numbered exercises
       ("In the following exercises, simplify.", "For Exercises 5-6, …") — must be tagged.
       A node that BEGINS WITH ITS OWN EXERCISE NUMBER is an exercise, never a lead-in,
       even if its own imperative reads like an instruction. Prose/headers are not lead-ins.

    Return correct=True only if the splits and the tagged positions are BOTH right for
    this window (nothing missed, nothing spurious). Otherwise correct=False and say why.
    """

    current_nodes: list = dspy.InputField(
        description="The window's nodes, each {position, type, content}."
    )
    splits: list = dspy.InputField(
        description="Predicted splits: each {position, exercises:[{number, content}]}."
    )
    instruction_positions: list = dspy.InputField(description="Predicted lead-in node positions.")
    reason: str = dspy.OutputField(
        description="Brief justification; name any missed/spurious split or tag."
    )
    correct: bool = dspy.OutputField(
        description="True iff both the splits and the tags are correct for this window."
    )


def _judge() -> dspy.Module:
    """Cached judge module bound to the strong teacher model."""
    global _JUDGE
    try:
        return _JUDGE
    except NameError:
        pass
    j = dspy.ChainOfThought(JudgeSplit)
    j.set_lm(teacher_lm())
    _JUDGE = j
    return j


def _dump(items):
    """Normalise pydantic objects / dicts to plain JSON-able lists for the judge."""
    out = []
    for it in items or []:
        out.append(it.model_dump() if hasattr(it, "model_dump") else it)
    return out


def splitter_score(example, pred, trace=None) -> bool:
    """DSPy metric: True iff the strong judge approves the prediction for this window.

    Reference-free — uses ``example.current_nodes`` (the input the splitter saw) and the
    module's ``pred``; no gold labels needed. Returned bool gates whether a bootstrapped
    trace becomes a few-shot demonstration.
    """
    r = _judge()(
        current_nodes=_dump(getattr(example, "current_nodes", None)),
        splits=_dump(getattr(pred, "splits", None)),
        instruction_positions=list(getattr(pred, "instruction_positions", None) or []),
    )
    return bool(r.correct)
