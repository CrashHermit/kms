import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType
from .llm import text_lm


def _strip_orphan_label_separator(content: str) -> str:
    """Drop a leading separator left behind when only the number was removed.

    The refiner is asked to strip an exercise's number label; models sometimes
    remove just the digits (e.g. ``44``) and leave the trailing ``.``/``)`` and
    space, so the content comes back as ``. $x$``. This removes a single such
    orphan separator so the exercise starts cleanly, while leaving genuine
    content (e.g. a leading ``(a)`` subpart) untouched.
    """
    stripped = content.lstrip()
    if stripped[:1] in (".", ")"):
        stripped = stripped[1:].lstrip()
    return stripped


class Signature(dspy.Signature):
    """
    You are refining a single student exercise extracted from a technical textbook.

    Two jobs:
    1. Pull out the exercise's own number label — the identifier the book uses to
       refer to this problem (e.g. `12`, `12a`, `3.4`, `(iv)`). Return it exactly as
       written, as a string, with no surrounding punctuation (drop a trailing `.` or
       `)`, keep intrinsic parts like a subpart letter). Return None if the exercise
       carries no visible number label.
    2. Return the exercise's content with that number label removed and nothing else
       changed. Remove the whole label token — the digits AND any trailing
       separator such as `.` or `)` and the whitespace after it — so the content
       does not start with a dangling `.` or `)`. Preserve every other character
       verbatim — prose, LaTeX math (`$ $` inline, `$$ $$` display), subparts, and
       any attached material — exactly as given. Only the label is stripped; if
       there is no label, return the content unchanged.

    Do not renumber, normalise, invent, solve, or reword — extract the label and
    remove it, nothing more.
    """

    exercise_content: str = dspy.InputField(
        description="The full markdown content of a single exercise node."
    )

    number: str | None = dspy.OutputField(
        description="The exercise's own number label as a string (e.g. '12', '12a'), or None if it has none."
    )

    cleaned_content: str = dspy.OutputField(
        description="The exercise content with its own number label removed and everything else preserved verbatim."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.refiner = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, exercise_content: str):
        result = await self.refiner.acall(exercise_content=exercise_content)
        cleaned = result.cleaned_content
        if cleaned:
            cleaned = _strip_orphan_label_separator(cleaned)
        return dspy.Prediction(number=result.number, cleaned_content=cleaned)


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
        """Fill `number` and strip the label from `content` on every exercise node in one segment."""
        segment: Segment = state["segment"]
        for node in segment.nodes:
            if node.type == NodeType.EXERCISE and node.content:
                prediction = await self.module.aforward(exercise_content=node.content)
                node.number = prediction.number
                node.content = prediction.cleaned_content
        return {"exercise_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's refined nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("exercise_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
