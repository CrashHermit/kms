import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType
from .llm import text_lm


class Signature(dspy.Signature):
    r"""
    You are rewriting a single student exercise so that it stands on its own.

    The exercise was written under a shared lead instruction that applied to a whole
    group of exercises (e.g. "Find the derivative of the function."). That
    instruction is provided separately, and the exercise's own number label has
    already been removed from its content.

    Your job: rewrite the exercise's content so the instruction is woven into it
    naturally, producing a single self-contained problem statement a student could
    read in isolation.

    RULES:
    - Preserve the exercise's own substance exactly — its expressions, values,
      subparts, and any attached material. Do not solve it, and do not add or drop
      information.
    - Integrate the instruction as natural prose. You need not prepend it verbatim if
      that reads awkwardly, but never change its meaning.
    - All mathematical notation stays in LaTeX (`$ $` inline, `$$ $$` display).
    """

    instruction: str = dspy.InputField(
        description="The shared lead instruction governing this exercise."
    )
    exercise_content: str = dspy.InputField(
        description="The exercise's own content, with its number label already removed."
    )

    content: str = dspy.OutputField(
        description="The rewritten, self-contained exercise with the instruction woven in naturally."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.distributor = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, instruction: str, exercise_content: str):
        result = await self.distributor.acall(
            instruction=instruction,
            exercise_content=exercise_content,
        )
        return dspy.Prediction(content=result.content)


# --- LangGraph node: weave each instruction into the exercises it governs ---
#
# Governance is resolved by position, not by number alone: an instruction owns the
# run of exercises that follows it, up to the next instruction or header. Exercise
# numbers reset per section, so the same label can recur elsewhere in the document;
# the positional walk keeps an instruction's range from reaching across a boundary.
# The resolution runs in `dispatch` (single-threaded, whole-backbone view); workers
# only apply the already-resolved rewrites, so each exercise is written exactly once.

class InstructionDistributorNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Resolve exercise -> governing instruction across the document, then fan out per segment."""
        segments = state.get("segments", [])
        active = None  # the instruction node currently governing the run

        # seg.index -> list of (node position in seg.nodes, instruction content)
        assignments: dict[int, list[tuple[int, str]]] = {}
        for seg in segments:
            for pos, node in enumerate(seg.nodes):
                if node.type == NodeType.HEADER:
                    active = None
                elif node.type == NodeType.INSTRUCTION:
                    active = node
                elif node.type == NodeType.EXERCISE:
                    if (
                        active is not None
                        and node.number is not None
                        and node.number in active.exercise_numbers
                    ):
                        assignments.setdefault(seg.index, []).append((pos, active.content))

        sends = [
            Send("instruction_distributor_worker", {"segment": seg, "assignments": assignments[seg.index]})
            for seg in segments
            if seg.index in assignments
        ]
        return sends or "instruction_distributor_collect"

    async def worker(self, state: dict) -> dict:
        """Weave the governing instruction into each assigned exercise in one segment."""
        segment: Segment = state["segment"]
        assignments: list[tuple[int, str]] = state["assignments"]
        for pos, instruction in assignments:
            node = segment.nodes[pos]
            if node.content:
                prediction = await self.module.aforward(
                    instruction=instruction,
                    exercise_content=node.content,
                )
                node.content = prediction.content
        return {"distribute_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's rewritten nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("distribute_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
