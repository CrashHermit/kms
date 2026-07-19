import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType


class Signature(dspy.Signature):
    """
    You are refining a shared lead instruction that governs a group of student
    exercises in a technical textbook (e.g. `1-20 Find the derivative of the
    function.`).

    Your only job is to read the instruction's content, find the exercise range it
    covers, and expand that range into a flat list of individual exercise number
    labels, as strings, in order.

    RULES:
    - Enumerate every number in the range explicitly. `1-20` becomes
      `["1", "2", "3", ..., "20"]` — list all twenty, do not abbreviate or skip.
    - Handle compound ranges: `1, 3, 5-9` becomes `["1", "3", "5", "6", "7", "8", "9"]`.
    - Emit each label as a string exactly as it would be numbered.
    - Return an empty list if the instruction states no exercise range.

    Do not include the instruction's prose, and do not invent numbers beyond the
    stated range.
    """

    instruction_content: str = dspy.InputField(
        description="The full markdown content of a single instruction node."
    )

    exercise_numbers: list[str] = dspy.OutputField(
        description="The flat, fully-enumerated list of exercise number labels this instruction governs, as strings in order."
    )


class Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)

    async def aforward(self, instruction_content: str):
        result = await self.refiner.acall(instruction_content=instruction_content)
        return dspy.Prediction(exercise_numbers=result.exercise_numbers)


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
        """Fill `exercise_numbers` on every instruction node in one segment."""
        segment: Segment = state["segment"]
        for node in segment.nodes:
            if node.type == NodeType.INSTRUCTION and node.content:
                prediction = await self.module.aforward(instruction_content=node.content)
                node.exercise_numbers = prediction.exercise_numbers
        return {"instruction_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's refined nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("instruction_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
