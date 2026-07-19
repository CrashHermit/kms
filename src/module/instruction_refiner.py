import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType


class Signature(dspy.Signature):
    """
    You are refining a shared lead instruction that governs a group of student
    exercises in a technical textbook (e.g. `1-20 Find the derivative of the
    function.`).

    Two jobs:
    1. Find the exercise range the instruction covers and expand it into a flat list
       of individual exercise number labels, as strings, in order:
       - Enumerate every number explicitly. `1-20` becomes
         `["1", "2", "3", ..., "20"]` — list all twenty, do not abbreviate or skip.
       - Handle compound ranges: `1, 3, 5-9` becomes `["1", "3", "5", "6", "7", "8", "9"]`.
       - Return an empty list if the instruction states no exercise range.
    2. Return the instruction's content with that range removed and nothing else
       changed — leaving just the instruction prose (e.g. `Find the derivative of
       the function.`). Preserve all other text and LaTeX (`$ $` inline, `$$ $$`
       display) verbatim; if there is no range, return the content unchanged.

    Do not include exercise content, invent numbers beyond the stated range, or
    reword the instruction beyond removing the range.
    """

    instruction_content: str = dspy.InputField(
        description="The full markdown content of a single instruction node."
    )

    exercise_numbers: list[str] = dspy.OutputField(
        description="The flat, fully-enumerated list of exercise number labels this instruction governs, as strings in order."
    )

    cleaned_content: str = dspy.OutputField(
        description="The instruction content with the exercise range removed and everything else preserved verbatim."
    )


class Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)

    async def aforward(self, instruction_content: str):
        result = await self.refiner.acall(instruction_content=instruction_content)
        return dspy.Prediction(
            exercise_numbers=result.exercise_numbers,
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
                node.exercise_numbers = prediction.exercise_numbers
                node.content = prediction.cleaned_content
        return {"instruction_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's refined nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("instruction_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
