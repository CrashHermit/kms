"""
LLM-as-judge metric for the instruction-distributor DSPy program.

The distributor's one hard decision is the *extent* of governance: given a lead-in and
the problems that follow it, which run does its shared instruction cover, and where does
that run stop? The judge (on the strong teacher, deepseek-v4-pro) grades that decision
reference-free — so the trainset needs inputs only, no gold labels.

This is where the stress test found the soft failure (the Lebl 2.1.11 over-extension:
a "find the limit" lead-in stamped onto a "finish the proof" exercise). The judge is
written to catch exactly that — over-extension onto a differently-tasked problem, an
early stop, or a wrong/unstripped instruction string.
"""

import dspy

from kms.core.llm import teacher_lm


class JudgeGovern(dspy.Signature):
    r"""
    You are grading an instruction DISTRIBUTOR. It is given an exercise LEAD-IN and the
    problems that FOLLOW it in document order (each with a local `position`), and it must
    decide which of them the lead-in's shared instruction governs, plus the instruction.

    Correct behaviour:
      * governed_positions — a run that STARTS at the first following problem (position 0)
        and continues while the lead-in's imperative still sensibly applies, then STOPS the
        moment a problem is a clearly different task or the start of a different group. Judge
        by MEANING, not by numbering. A "find/compute" lead-in does NOT govern a following
        "prove/finish the proof" problem; a drill lead-in does NOT govern a standalone word
        problem. Over-extending past such a boundary is WRONG; stopping early is WRONG.
      * instruction — the shared imperative, copied WITHOUT any "In Exercises X-Y," /
        "In the following exercises," framing; empty iff nothing is governed.

    Return correct=True only if BOTH governed_positions and instruction are right for this
    lead-in and these following problems. Otherwise correct=False and name the error.
    """

    lead_in: str = dspy.InputField(description="The lead-in node's text.")
    following_problems: list = dspy.InputField(
        description="Following problems, each {position, number, text}."
    )
    instruction: str = dspy.InputField(description="Predicted shared instruction (may be empty).")
    governed_positions: list = dspy.InputField(
        description="Predicted governed positions (a run from 0)."
    )
    reason: str = dspy.OutputField(
        description="Brief justification; name any over-extension, early stop, or bad instruction."
    )
    correct: bool = dspy.OutputField(
        description="True iff governed_positions and instruction are both correct here."
    )


def _judge() -> dspy.Module:
    global _JUDGE
    try:
        return _JUDGE
    except NameError:
        pass
    j = dspy.ChainOfThought(JudgeGovern)
    j.set_lm(teacher_lm())
    _JUDGE = j
    return j


def _dump(items):
    out = []
    for it in items or []:
        out.append(it.model_dump() if hasattr(it, "model_dump") else it)
    return out


def distributor_score(example, pred, trace=None) -> bool:
    """DSPy metric: True iff the strong judge approves the governance decision.

    Reference-free — uses ``example.lead_in`` / ``example.following_problems`` (the input the
    distributor saw) and the module's ``pred``. Gates whether a bootstrapped trace becomes a
    few-shot demonstration.
    """
    r = _judge()(
        lead_in=getattr(example, "lead_in", "") or "",
        following_problems=_dump(getattr(example, "following_problems", None)),
        instruction=(getattr(pred, "instruction", None) or ""),
        governed_positions=list(getattr(pred, "governed_positions", None) or []),
    )
    return bool(r.correct)
