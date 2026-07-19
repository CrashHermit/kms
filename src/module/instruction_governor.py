r"""
Instruction governance.

A shared lead instruction ("For the following exercises, find the amplitude...")
governs a run of exercises. Rather than rewrite each exercise to fuse the lead in,
this stage decides *which* exercises a lead governs and records the lead on each
governed exercise's ``instruction`` field — the exercise content is never touched.
Presenting a governed exercise as self-contained (lead + content) is then a
rendering concern for the consumer, not a destructive pipeline transform.

Governance is positional + semantic. Each instruction opens a group of the
exercises that follow it, up to the next instruction or header — an intentionally
over-inclusive pool. One batched LLM call per group judges which exercises the lead
actually governs; a self-contained ``[T]`` word problem sitting in the pool comes
back False. Batching (whole pool in one call) is both cheaper than per-exercise and
more accurate: the contrast between terse governed exercises and self-contained ones
is itself the signal. Large pools split into POOL_CAP-sized batches so the boolean
list stays aligned even for a small model.
"""

import dspy
from langgraph.types import Send

from .state import State, NodeType
from .llm import text_lm

# Max exercises judged in one call. Keeps the boolean list short enough to stay
# aligned; a larger group fans out into several batches sharing the same lead.
POOL_CAP = 15


class GovernanceSig(dspy.Signature):
    r"""
    Decide which exercises a shared lead instruction governs.

    A lead like "For the following exercises, find the amplitude, period, and phase
    shift." governs terse exercises that need it to become a complete task (e.g.
    "$y = 3\cos(2x+3)$"), but does NOT govern a self-contained problem that already
    states its own task (e.g. a "[T]" application word problem like "The diameter of
    a wheel rolling on the ground is 40 in...").

    You are given the lead and the exercises that follow it, in order. For EACH
    exercise decide whether the lead governs it: True if the exercise is incomplete
    without the lead, False if it stands on its own.

    Return `governed`: a list of booleans, one per exercise, in the same order. The
    list length MUST equal the number of exercises given.
    """

    instruction: str = dspy.InputField(description="The shared lead instruction.")
    exercises: list[str] = dspy.InputField(
        description="The exercise contents that follow the lead, in document order."
    )
    governed: list[bool] = dspy.OutputField(
        description="One boolean per exercise, in order: True if the lead governs that exercise."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.judge = dspy.ChainOfThought(GovernanceSig)
        self.set_lm(lm or text_lm())

    async def aforward(self, instruction: str, exercises: list[str]) -> list[bool]:
        result = await self.judge.acall(instruction=instruction, exercises=exercises)
        governed = list(result.governed or [])
        # Length guardrail: on any mismatch, pad/truncate to the input length and
        # default the uncertain ones to un-governed rather than misalign the mapping.
        if len(governed) != len(exercises):
            governed = (governed + [False] * len(exercises))[: len(exercises)]
        return [bool(g) for g in governed]


# --- LangGraph node: annotate each governed exercise with its lead ---

class InstructionGovernorNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Build instruction-anchored groups across the flat node stream and fan out
        one Send per group (splitting oversized pools into POOL_CAP batches)."""
        segments = state.get("segments", [])
        lead: str | None = None
        members: list[tuple[int, int, str]] = []  # (seg_index, pos, content)
        sends: list[Send] = []

        def flush() -> None:
            if lead and members:
                for i in range(0, len(members), POOL_CAP):
                    sends.append(Send("governor_worker", {"instruction": lead, "members": members[i:i + POOL_CAP]}))

        for seg in segments:
            for pos, node in enumerate(seg.nodes):
                if node.type == NodeType.HEADER:
                    flush()
                    lead, members = None, []
                elif node.type == NodeType.INSTRUCTION:
                    flush()
                    lead, members = (node.content or None), []
                elif node.type == NodeType.EXERCISE and lead and node.content:
                    members.append((seg.index, pos, node.content))
        flush()
        return sends or "governor_collect"

    async def worker(self, state: dict) -> dict:
        """Judge which exercises in one group the lead governs; emit (seg, pos, lead)
        for the governed ones."""
        instruction: str = state["instruction"]
        members: list[tuple[int, int, str]] = state["members"]
        governed = await self.module.aforward(instruction, [content for _s, _p, content in members])
        results = [
            (seg_index, pos, instruction)
            for (seg_index, pos, _content), keep in zip(members, governed)
            if keep
        ]
        return {"governance_results": results}

    def collect(self, state: State) -> dict:
        """Write the governing lead onto each governed exercise's `instruction` field."""
        by_index = {seg.index: seg for seg in state["segments"]}
        for seg_index, pos, instruction in state.get("governance_results", []):
            seg = by_index.get(seg_index)
            if seg is not None and 0 <= pos < len(seg.nodes):
                seg.nodes[pos].instruction = instruction
        return {"segments": state["segments"]}
