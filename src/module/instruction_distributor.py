r"""
Instruction distributor — copies a grouped-exercise lead-in's shared directive onto the
Problem entities it governs, by a growing-window walk (no number/range parsing).

A run of exercises often states its imperative ONCE, in a lead-in that governs the whole run.
That lead-in is not a member of any individual problem, so the per-entity Problem attributor
can't see it — AutoMathKG's `instruction` is inherently a cross-entity, positional attribute.
The splitter has already TAGGED each lead-in node `role="instruction"`; this pass reads those
tags and distributes the directive.

The extent is judged by the LLM, not by numbers. A lead-in may name a range ("In Exercises
1.23-1.25, …") or may not ("Answer the following.", "Prove each of the following."), so a range
parser can't decide who is governed — the model reads the following problems and decides where
the governed run ends. It is the SAME growing look-ahead used by the finders and the splitter:

  * Anchor on a tagged lead-in node. Its candidate problems are the ones that follow it, up to
    the next lead-in (a new lead-in starts a new governance).
  * Take a look-ahead window of those following problems (whole problems up to a token budget)
    and ask the LLM which of them the lead-in governs, plus the shared instruction to apply.
  * If the governed run reaches the window's edge it may continue, so GROW the window (double
    the budget) and re-read; if a non-governed problem is seen to follow the run (bounded), or
    the candidates are exhausted, BANK — stamp `entity.instruction` on the governed problems.

It runs AFTER the Problem attributor (it reads each problem's `contents`/`number` to judge
governance). Entry point ``distribute_instructions(nodes, problems, module)`` mutates the
problems in place; ``InstructionDistributorNode`` wires it onto the ``problem_entities`` channel.
"""

import dspy
from pydantic import BaseModel, Field

from .state import State, ASTNode, Entity
from .llm import text_lm

# Same growing look-ahead shape as the finders/splitter (~4 chars/token).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


class WindowProblem(BaseModel):
    """One following problem as the LLM sees it: a local position, its number, its statement."""
    position: int
    number: str | None = None
    text: str | None = None


class GovernExtent(dspy.Signature):
    r"""
    Given an exercise LEAD-IN and the problems that FOLLOW it in document order, decide which of
    those problems the lead-in's shared instruction governs, and give the instruction to apply.

    A lead-in states one imperative for a run of exercises. Some name a range ("In Exercises
    1.23-1.25, find the eigenvalues of each matrix."), some do not ("Answer the following.",
    "Prove each of the following statements."). Judge governance by MEANING, not by numbers: the
    governed problems are a run that STARTS at the first following problem and continues while
    the lead-in's instruction still sensibly applies to them, and STOPS when it no longer does —
    a problem that is clearly a different task, or the start of a different group.

    Return:
      * instruction — the shared imperative to apply to the governed problems, copied as written
        but WITHOUT any "In Exercises X-Y," framing ("find the eigenvalues of each matrix",
        "answer the following"). EMPTY string if the lead-in actually governs nothing here.
      * governed_positions — the `position` values of the governed problems (a run from the
        first). EMPTY list if none are governed.
    """

    lead_in: str = dspy.InputField(description="The lead-in node's text.")
    following_problems: list[WindowProblem] = dspy.InputField(
        description="The problems that follow the lead-in, in order, each with a local position."
    )
    instruction: str = dspy.OutputField(description="The shared imperative to apply, without range framing, or empty string.")
    governed_positions: list[int] = dspy.OutputField(description="Positions of the governed problems, a run from the first; empty if none.")


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.judge = dspy.ChainOfThought(GovernExtent)
        self.set_lm(lm or text_lm())

    async def govern(self, lead_in: str, following: list[WindowProblem]) -> tuple[str, list[int]]:
        r = await self.judge.acall(lead_in=lead_in, following_problems=following)
        return (r.instruction or "").strip(), list(r.governed_positions or [])


def _est_tokens(problem: Entity) -> int:
    return len(_problem_text(problem)) // 4 + 1


def _problem_text(problem: Entity) -> str:
    """The problem's statement as the LLM should see it — its attributed contents, or its
    number as a last resort."""
    body = " ".join(c for c in (problem.contents or []) if c)
    return body or (problem.number or "")


def _window(candidates: list[Entity], budget: int) -> list[Entity]:
    """Whole following problems up to the soft token budget, always at least one."""
    window, tokens = [], 0
    for problem in candidates:
        t = _est_tokens(problem)
        if window and tokens + t > budget:
            break
        window.append(problem)
        tokens += t
    return window


async def _govern_one(node: ASTNode, candidates: list[Entity], module: Module) -> None:
    """Growing-window walk for one lead-in: find the governed run among its following problems
    and stamp the instruction on them. Grows the window while the run reaches its edge."""
    if not candidates:
        return
    size = LOOKAHEAD_BUDGET
    while True:
        window = _window(candidates, size)
        last_local = len(window) - 1
        exhausted = len(window) == len(candidates)

        instruction, positions = await module.govern(
            node.content or "",
            [WindowProblem(position=k, number=window[k].number, text=_problem_text(window[k]))
             for k in range(len(window))],
        )
        governed = sorted({min(max(p, 0), last_local) for p in positions})

        if not governed:
            return  # this lead-in governs nothing here
        run_end = governed[-1]

        if exhausted or size >= MAX_LOOKAHEAD_BUDGET or run_end < last_local:
            # Bounded (a non-governed problem follows the run), or nothing left to gather: bank.
            if instruction:
                for k in governed:
                    window[k].instruction = instruction
            return
        size *= 2  # the run reaches the window edge and may continue — grow and re-read


async def distribute_instructions(
    nodes: list[ASTNode],
    problems: list[Entity],
    module: Module | None = None,
) -> list[Entity]:
    """Stamp each governed Problem's `instruction` from the tagged lead-in nodes (in place),
    judging the governed run per lead-in with a growing-window LLM walk (no number matching)."""
    lead_ins = [n for n in nodes if n.role == "instruction"]
    if not lead_ins or not problems:
        return problems
    module = module or Module()
    order = {n.id: i for i, n in enumerate(nodes)}
    far = len(order)

    def pos(entity: Entity) -> int:
        return order.get(entity.members[0], far) if entity.members else far

    ordered = sorted(problems, key=pos)
    lead_positions = sorted(order.get(n.id, -1) for n in lead_ins)

    for node in lead_ins:
        here = order.get(node.id, -1)
        # A new lead-in starts a new governance, so a lead-in's candidates end at the next one.
        nxt = min((p for p in lead_positions if p > here), default=far)
        candidates = [p for p in ordered if here < pos(p) < nxt]
        await _govern_one(node, candidates, module)
    return problems


# --- LangGraph node: distribute instructions over the attributed Problem entities ---

class InstructionDistributorNode:
    """Stamps `instruction` onto the governed Problems, reading the splitter's `role="instruction"`
    lead-in tags. Runs after the Problem attributor (it reads each problem's contents/number) and
    writes the enriched entities back to the ``problem_entities`` channel."""

    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        problems = state.get("problem_entities", [])
        await distribute_instructions(state.get("nodes", []), problems, module=self.module)
        return {"problem_entities": problems}
