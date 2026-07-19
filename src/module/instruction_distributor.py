import dspy
from langgraph.types import Send

from .state import State, Segment, NodeType
from .llm import text_lm


class Signature(dspy.Signature):
    r"""
    You are rewriting a single student exercise so that it stands on its own.

    A shared lead instruction that applied to a whole group of exercises (e.g.
    "Find the derivative of the function.") is provided separately. The exercise's
    own number label has already been removed from its content.

    Two jobs:

    1. APPLICABILITY — decide whether the shared instruction actually governs THIS
       exercise. A terse shared lead ("Find the amplitude, period, and phase shift.")
       governs terse exercises of that kind (e.g. "$y = 3\cos(2x+3)$"), but does NOT
       govern a self-contained problem that already states its own task (e.g. a `[T]`
       word problem like "The diameter of a wheel rolling on the ground is 40 in...").
       Set `applies` to true only when the instruction genuinely completes this
       exercise's task.

    2. REWRITE — if it applies, rewrite the exercise so the instruction is woven in
       naturally, producing a single self-contained problem statement. If it does
       NOT apply, set applies=false and return the exercise content unchanged.

    RULES:
    - Preserve the exercise's own substance exactly — its expressions, values,
      subparts, and any attached material. Do not solve it, and do not add or drop
      information.
    - Integrate the instruction as natural prose. You need not prepend it verbatim if
      that reads awkwardly (drop plural framing like "For the following exercises,"),
      but never change its meaning.
    - All mathematical notation stays in LaTeX (`$ $` inline, `$$ $$` display).
    """

    instruction: str = dspy.InputField(
        description="The shared lead instruction that may govern this exercise."
    )
    exercise_content: str = dspy.InputField(
        description="The exercise's own content, with its number label already removed."
    )

    applies: bool = dspy.OutputField(
        description="True if the shared instruction genuinely governs this exercise; False if the exercise is self-contained and the instruction does not apply."
    )
    content: str = dspy.OutputField(
        description="If applies, the rewritten self-contained exercise with the instruction woven in; if not, the original exercise content unchanged."
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
        return dspy.Prediction(applies=result.applies, content=result.content)


# --- LangGraph node: weave each instruction into the exercises it governs ---
#
# Governance is positional first: an instruction owns the run of exercises that
# follows it, up to the next instruction or header. Exercise numbers reset per
# section, so this positional scoping keeps a range from reaching across a section.
# Within that window two cases decide membership:
#
#   * The instruction states a numeric range (e.g. "1-20"): weave an exercise only
#     if its number falls in the range. This deterministically drops free-floating
#     orphans — "1-20 do X", then 21 and 22 with no lead, then "23-40 do Y" — since
#     21/22 are outside every range. (needs_check=False: trust the range.)
#   * The instruction states no range (e.g. "For the following exercises, ..."): the
#     exercise is only a candidate; the distributor's applicability judgment decides
#     whether the lead actually governs it, so a self-contained [T] problem that
#     merely follows the lead is left alone. (needs_check=True.)
#
# An in-range window with a missing exercise number also defers to the judgment
# rather than being silently dropped. Resolution runs in `dispatch` (single-threaded,
# whole-backbone view); workers only apply the resolved rewrites.

class InstructionDistributorNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Resolve exercise -> governing instruction across the document, then fan out per segment."""
        segments = state.get("segments", [])
        active = None  # the instruction node currently governing the run

        # seg.index -> list of (node position, instruction content, needs_check)
        assignments: dict[int, list[tuple[int, str, bool]]] = {}
        for seg in segments:
            for pos, node in enumerate(seg.nodes):
                if node.type == NodeType.HEADER:
                    active = None
                elif node.type == NodeType.INSTRUCTION:
                    active = node
                elif node.type == NodeType.EXERCISE and active is not None and node.content:
                    if active.exercise_numbers:
                        # Range present: trust the numbers to bound governance.
                        if node.number is not None and node.number in active.exercise_numbers:
                            assignments.setdefault(seg.index, []).append((pos, active.content, False))
                        elif node.number is None:
                            # Can't confirm membership — let the judgment decide.
                            assignments.setdefault(seg.index, []).append((pos, active.content, True))
                        # else: number present but out of range — genuine orphan, skip.
                    else:
                        # No range: positional candidate, gated by applicability.
                        assignments.setdefault(seg.index, []).append((pos, active.content, True))

        sends = [
            Send("instruction_distributor_worker", {"segment": seg, "assignments": assignments[seg.index]})
            for seg in segments
            if seg.index in assignments
        ]
        return sends or "instruction_distributor_collect"

    async def worker(self, state: dict) -> dict:
        """Weave the governing instruction into each assigned exercise in one segment."""
        segment: Segment = state["segment"]
        assignments: list[tuple[int, str, bool]] = state["assignments"]
        for pos, instruction, needs_check in assignments:
            node = segment.nodes[pos]
            if not node.content:
                continue
            prediction = await self.module.aforward(
                instruction=instruction,
                exercise_content=node.content,
            )
            # Range-confirmed assignments weave unconditionally; candidates only if
            # the distributor judges the instruction actually applies.
            if needs_check and not prediction.applies:
                continue
            node.content = prediction.content
        return {"distribute_results": [(segment.index, segment.nodes)]}

    def collect(self, state: State) -> dict:
        """Merge each segment's rewritten nodes back into the ordered backbone."""
        nodes_by_index = dict(state.get("distribute_results", []))
        for segment in state["segments"]:
            if segment.index in nodes_by_index:
                segment.nodes = nodes_by_index[segment.index]
        return {"segments": state["segments"]}
