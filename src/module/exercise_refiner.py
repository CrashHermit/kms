import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType
from .llm import text_lm


def _clean_number(number: str | None) -> str | None:
    """Strip stray surrounding quotes/whitespace off an extracted number label.

    DeepSeek's JSON-mode output sometimes wraps a short string field's value in
    literal quotes (``"113"`` instead of ``113``), which would break the range
    membership check downstream. Normalise it here; an empty result becomes None.
    """
    if number is None:
        return None
    cleaned = number.strip().strip("\"'").strip()
    # v4-flash JSON mode sometimes writes the string "null"/"None" instead of a real
    # null; normalise those (and empty) to None so the label stays clean metadata.
    if cleaned.lower() in ("", "null", "none", "nan"):
        return None
    return cleaned


class Signature(dspy.Signature):
    """
    You are reading a single student exercise extracted from a technical textbook.

    Pull out the exercise's own number label — the identifier the book uses to refer
    to this problem (e.g. `12`, `12a`, `3.4`, `(iv)`). Return it exactly as written,
    as a string, with no surrounding punctuation (drop a trailing `.` or `)`, keep
    intrinsic parts like a subpart letter). Return None if the exercise carries no
    visible number label.

    Do not renumber, normalise, invent, solve, reword, or otherwise change anything —
    only report the label. The exercise content is left exactly as-is by the caller.
    """

    exercise_content: str = dspy.InputField(
        description="The full markdown content of a single exercise node."
    )

    number: str | None = dspy.OutputField(
        description="The exercise's own number label, e.g. 12 or 12a (no surrounding quotes), or None if it has none."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, exercise_content: str):
        result = await self.refiner.acall(exercise_content=exercise_content)
        return dspy.Prediction(number=_clean_number(result.number))


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
        """Record each exercise node's `number` label; content is left untouched."""
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
