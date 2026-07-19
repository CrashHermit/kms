import re

import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType
from .llm import text_lm

# Hyphen, en dash, or em dash between two numeric range bounds.
_RANGE_SEP = re.compile(r"\s*[-–—]\s*")


def parse_exercise_range(spec: str | None) -> list[str]:
    """Expand a compact exercise-range token into a flat list of number labels.

    Deterministic in code rather than asked of the model — LLMs miscount long
    ranges. Handles single numbers and numeric ranges separated by commas:
    ``"1-20"`` -> ``["1", ..., "20"]``, ``"1, 3, 5-9"`` -> ``["1","3","5",...,"9"]``.
    Unparseable parts are skipped so a messy token can never crash the stage.
    """
    if not spec:
        return []
    spec = spec.strip().strip("\"'").strip()  # tolerate JSON-mode over-quoting
    labels: list[str] = []
    for part in spec.split(","):
        part = part.strip().strip("\"'").strip()
        if not part:
            continue
        bounds = _RANGE_SEP.split(part)
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            lo, hi = int(bounds[0]), int(bounds[1])
            if lo <= hi and hi - lo <= 500:  # guard against an absurd expansion
                labels.extend(str(n) for n in range(lo, hi + 1))
        elif part.isdigit():
            labels.append(part)
        # else: non-numeric / malformed token — skip it
    return labels


class Signature(dspy.Signature):
    """
    You are refining a shared lead instruction that governs a group of student
    exercises in a technical textbook (e.g. `1-20 Find the derivative of the
    function.`).

    Two jobs:
    1. Extract the exercise-range token the instruction states — the raw range as
       written, NOT an enumerated list. Examples:
       - `1-20 Find the derivative...` -> `"1-20"`
       - `1, 3, 5-9 Evaluate...` -> `"1, 3, 5-9"`
       - `For the following exercises, convert...` (no range) -> None
       Return only the compact numeric token (commas and dashes are fine); do not
       expand it, and do not invent a range that is not written.
    2. Return the instruction's content with that range token removed and nothing
       else changed — leaving just the instruction prose (e.g. `Find the derivative
       of the function.`). Preserve all other text and LaTeX (`$ $` inline, `$$ $$`
       display) verbatim; if there is no range, return the content unchanged.

    Do not include exercise content or reword the instruction beyond removing the range.
    """

    instruction_content: str = dspy.InputField(
        description="The full markdown content of a single instruction node."
    )

    exercise_range: str | None = dspy.OutputField(
        description="The raw exercise-range token as written (e.g. '1-20' or '1, 3, 5-9'), or None if the instruction states no range."
    )

    cleaned_content: str = dspy.OutputField(
        description="The instruction content with the exercise range removed and everything else preserved verbatim."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, instruction_content: str):
        result = await self.refiner.acall(instruction_content=instruction_content)
        return dspy.Prediction(
            exercise_range=result.exercise_range,
            cleaned_content=result.cleaned_content,
        )


# --- LangGraph node: expand each instruction's range into flat exercise labels ---

class InstructionRefinerNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per segment that carries at least one instruction node."""
        segments = state.get("segments", [])
        sends = [
            Send("instruction_refiner_worker", {"segment": seg})
            for seg in segments
            if any(n.type == NodeType.INSTRUCTION for n in seg.nodes)
        ]
        return sends or "instruction_refiner_collect"

    async def worker(self, state: dict) -> dict:
        """Fill `exercise_numbers` and strip the range from `content` on every instruction node in one segment."""
        segment: Segment = state["segment"]
        for node in segment.nodes:
            if node.type == NodeType.INSTRUCTION and node.content:
                prediction = await self.module.aforward(instruction_content=node.content)
                node.exercise_numbers = parse_exercise_range(prediction.exercise_range)
                node.content = prediction.cleaned_content
        return {"instruction_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's refined nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("instruction_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
