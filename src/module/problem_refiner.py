import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType
from .llm import text_lm


def _clean_number(number: str | None) -> str | None:
    """Strip stray surrounding quotes/whitespace off an extracted number label.

    DeepSeek's JSON-mode output sometimes wraps a short string field's value in
    literal quotes (``"113"`` instead of ``113``) or writes the string ``"null"`` /
    ``"None"`` instead of a real null. Normalise those so the label stays clean
    metadata; an empty result becomes None.
    """
    if number is None:
        return None
    cleaned = number.strip().strip("\"'").strip()
    if cleaned.lower() in ("", "null", "none", "nan"):
        return None
    return cleaned


class Signature(dspy.Signature):
    """
    You are reading a single student problem extracted from a technical textbook.

    Pull out the problem's own number label — the identifier the book uses to refer
    to it (e.g. `12`, `12a`, `3.4`, `(iv)`). Return it exactly as written, as a
    string, with no surrounding punctuation (drop a trailing `.` or `)`, keep
    intrinsic parts like a subpart letter). Return None if the problem carries no
    visible number label.

    Do not renumber, normalise, invent, solve, reword, or otherwise change anything —
    only report the label. The problem content is left exactly as-is by the caller.
    """

    problem_content: str = dspy.InputField(
        description="The full markdown content of a single problem node."
    )

    number: str | None = dspy.OutputField(
        description="The problem's own number label, e.g. 12 or 12a (no surrounding quotes), or None if it has none."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, problem_content: str):
        result = await self.refiner.acall(problem_content=problem_content)
        return dspy.Prediction(number=_clean_number(result.number))


# --- LangGraph node: extract each problem's number label ---

class ProblemRefinerNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per segment that carries at least one problem node."""
        segments = state.get("segments", [])
        sends = [
            Send("problem_refiner_worker", {"segment": seg})
            for seg in segments
            if any(n.type == NodeType.PROBLEM for n in seg.nodes)
        ]
        return sends or "problem_refiner_collect"

    async def worker(self, state: dict) -> dict:
        """Record each problem node's `number` label; content is left untouched."""
        segment: Segment = state["segment"]
        for node in segment.nodes:
            if node.type == NodeType.PROBLEM and node.content:
                prediction = await self.module.aforward(problem_content=node.content)
                node.number = prediction.number
        return {"problem_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's refined nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("problem_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
