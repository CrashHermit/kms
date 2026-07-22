r"""
Instruction distributor — copies a grouped-exercise lead-in's shared directive onto the
Problem entities it governs.

A run of exercises often states its imperative ONCE, in a lead-in that governs the whole run
("In Exercises 1.23-1.25, find the eigenvalues of each matrix."). That lead-in is not a member
of any individual problem, so the per-entity Problem attributor can't see it — AutoMathKG's
`instruction` is inherently a cross-entity, positional attribute. The splitter has already
TAGGED each lead-in node `role="instruction"`; this pass reads those tags and distributes the
directive.

It is deliberately MINIMAL (MVP, range-only — no structural fallback):

  * For each tagged lead-in node, an LLM reads its text and returns the RANGE it governs
    (`start_number`..`end_number`, e.g. 1.23..1.25) and the shared `instruction` imperative.
  * If no explicit range is stated, the lead-in is SKIPPED — nothing is stamped. (A structural
    "govern until the next boundary" fallback is intentionally left out for now.)
  * The match is then deterministic: every Problem whose own `number` falls inside the range
    (dotted numbers compared component-wise, so 1.3 < 1.23) and which appears AFTER the lead-in
    in document order gets `entity.instruction` set to the directive.

It runs AFTER the Problem attributor, because it matches on each Problem's `number` — which the
attributor fills. Entry point ``distribute_instructions(nodes, problems, module)`` mutates the
problems in place; ``InstructionDistributorNode`` wires it onto the ``problem_entities`` channel.
"""

import dspy

from .state import State, ASTNode, Entity
from .llm import text_lm


class ExtractRange(dspy.Signature):
    r"""
    Read an exercise LEAD-IN (a directive that introduces a run of exercises) and extract the
    RANGE of exercise numbers it governs and the shared instruction.

    A lead-in names a span of exercises and states one imperative for all of them, e.g.
    "In Exercises 1.23-1.25, find the eigenvalues of each matrix." or "For Exercises 5 through
    8, determine whether the set is a subspace."

    Return:
      * start_number — the first exercise number in the range, as written ("1.23", "5"). EMPTY
        string if the lead-in does not state an explicit range.
      * end_number — the last exercise number in the range ("1.25", "8"). EMPTY if no range.
      * instruction — the shared imperative itself, copied as written but WITHOUT the "In
        Exercises X-Y," framing ("find the eigenvalues of each matrix"). EMPTY if there is none.
    """

    lead_in: str = dspy.InputField(description="The lead-in node's text.")
    start_number: str = dspy.OutputField(description="First exercise number in the governed range, or empty string.")
    end_number: str = dspy.OutputField(description="Last exercise number in the governed range, or empty string.")
    instruction: str = dspy.OutputField(description="The shared imperative, without the range framing, or empty string.")


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.extract = dspy.Predict(ExtractRange)
        self.set_lm(lm or text_lm())

    async def range_of(self, lead_in: str) -> tuple[str, str, str]:
        r = await self.extract.acall(lead_in=lead_in)
        return (r.start_number or "").strip(), (r.end_number or "").strip(), (r.instruction or "").strip()


def _parse_number(text: str | None) -> tuple[int, ...] | None:
    """A dotted reference number as a tuple of ints for order-correct comparison ("1.23" ->
    (1, 23), so 1.3 < 1.23). None if it isn't a plain dotted-integer number."""
    parts = (text or "").strip().split(".")
    if not parts or parts == [""]:
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


async def distribute_instructions(
    nodes: list[ASTNode],
    problems: list[Entity],
    module: Module | None = None,
) -> list[Entity]:
    """Stamp each governed Problem's `instruction` from the tagged lead-in nodes (in place).

    Range-only: a lead-in with no explicit numeric range is skipped. A Problem is governed when
    its `number` is inside the range AND it appears after the lead-in in document order."""
    lead_ins = [n for n in nodes if n.role == "instruction"]
    if not lead_ins or not problems:
        return problems
    module = module or Module()
    order = {n.id: i for i, n in enumerate(nodes)}
    far = len(order)

    for node in lead_ins:
        start_s, end_s, instruction = await module.range_of(node.content or "")
        start, end = _parse_number(start_s), _parse_number(end_s)
        if not (start and end and instruction):  # no explicit range -> skip (no fallback)
            continue
        lead_pos = order.get(node.id, -1)
        for problem in problems:
            num = _parse_number(problem.number)
            pos = order.get(problem.members[0], far) if problem.members else far
            if num is not None and start <= num <= end and pos > lead_pos:
                problem.instruction = instruction
    return problems


# --- LangGraph node: distribute instructions over the attributed Problem entities ---

class InstructionDistributorNode:
    """Stamps `instruction` onto the governed Problems, reading the splitter's `role="instruction"`
    lead-in tags. Runs after the Problem attributor (it matches on each Problem's `number`) and
    writes the enriched entities back to the ``problem_entities`` channel."""

    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    async def run(self, state: State) -> dict:
        problems = state.get("problem_entities", [])
        await distribute_instructions(state.get("nodes", []), problems, module=self.module)
        return {"problem_entities": problems}
