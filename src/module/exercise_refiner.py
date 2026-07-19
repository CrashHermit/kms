import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType


class Signature(dspy.Signature):
    """
    You are refining a single student exercise extracted from a technical textbook.

    Your only job is to read the exercise's content and pull out its own number
    label — the identifier the book uses to refer to this problem (e.g. `12`, `12a`,
    `3.4`, `(iv)`). Return it exactly as written, as a string, with no surrounding
    punctuation (drop a trailing `.` or `)`, keep intrinsic parts like a subpart
    letter).

    Return None if the exercise carries no visible number label.

    Do not renumber, normalise, or invent a label — transcribe only what is present.
    """

    exercise_content: str = dspy.InputField(
        description="The full markdown content of a single exercise node."
    )

    number: str | None = dspy.OutputField(
        description="The exercise's own number label as a string (e.g. '12', '12a'), or None if it has none."
    )


class Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)

    async def aforward(self, exercise_content: str):
        result = await self.refiner.acall(exercise_content=exercise_content)
        return dspy.Prediction(number=result.number)


# --- LangGraph node: extract each exercise's number label ---

class ExerciseRefinerNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per segment that carries at least one exercise node."""
        segments = state.get("segments", [])
        sends = [
            Send("exercise_refiner_worker", {"segment": seg})
            for seg in segments
            if any(n.type == NodeType.EXERCISE for n in seg.nodes)
        ]
        return sends or "exercise_refiner_collect"

    async def worker(self, state: dict) -> dict:
        """Fill `number` on every exercise node in one segment."""
        segment: Segment = state["segment"]
        for node in segment.nodes:
            if node.type == NodeType.EXERCISE and node.content:
                prediction = await self.module.aforward(exercise_content=node.content)
                node.number = prediction.number
        return {"exercise_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's refined nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("exercise_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
