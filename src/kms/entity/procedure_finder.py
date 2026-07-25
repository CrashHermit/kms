r"""
Procedure finder — one pass over every entity: "is there something to work out here, shown or absent?"

A procedure is **extracted from within an entity**, never detected as a block of its own. That is the
bi-modal spine of the graph: the entity is the declarative half (what is claimed or asked), the
procedure it yields is the procedural half (how it is established or solved), and the split between
them is *produced by extraction* rather than decided by a classifier
(``docs/GENERALIZATION.md``, "Entity layer").

This generalizes what the per-type attributors did as two hardcoded special cases — the theorem
attributor's ``proof_start`` and the problem attributor's ``solution_start`` — into one pass over any
entity, asking the question directly instead of deriving it from a type. That matters for a general
engine: a physics "derivation" and a biology "mechanism walkthrough" are the same shape as a proof,
and nothing in the routing has to learn their names.

Four outcomes, decided per entity:

* **shown** — the worked steps are on the page (a theorem's proof, an example's solution) →
  **extract**: split the statement from the worked part, decompose the worked part into role-labelled
  steps, attach it as a ``models.Procedure``.
* **absent, and the entity poses a task to solve** — an exercise with no worked solution →
  **create**: generate the steps (AutoMathKG-style completion, **task-first**), marked
  ``generated=True`` so the graph can always tell model output from page truth.
* **absent, and the entity asserts a claim** — a theorem with no shown proof → **defer**. Generating
  proofs is a different and much harder problem than solving a posed exercise; the entity keeps no
  procedure and the gap stays visible in the graph.
* **nothing to work out** — a definition, a bare statement of fact → **skip**.

Entry point ``find_procedure(entity, module)`` (async): writes ``entity.procedures`` and trims
``entity.contents`` to the statement half, in place, and returns the entity. Persistence-agnostic.
"""

import asyncio

import dspy
from pydantic import BaseModel

from kms.core import llm, models, state

# The role labels a step may carry — AutoMathKG's action taxonomy, minus `definition` (a derivation
# step establishes, it does not define). This stays closed on purpose: it labels the internal
# structure of one derivation, which the taxonomy covers across domains, unlike the cross-entity
# relation vocabulary that had to open up.
STEP_ACTIONS = [
    action for action in models.ACTIONS_ALL if action != 'definition'
]


class ContentPiece(BaseModel):
    """One piece of the entity's content as the assessment pass sees it: a position and its text."""

    position: int
    text: str


class Assess(dspy.Signature):
    r"""
    Read one textbook block and decide whether it contains — or calls for — a WORKED DERIVATION:
    an ordered argument or computation that establishes a claim or answers a question. A proof, a
    solution, a derivation, a calculation, a worked procedure.

    Answer four things:

      * has_work — is there anything here to work out AT ALL? A definition, a statement of notation,
        or a bare statement of fact has nothing to work out: answer false. A claim that must be
        established, or a question that must be answered, does: answer true.
      * shown — are the worked steps actually PRESENT in the text? True for a theorem followed by its
        proof, or an example followed by its solution. False for an exercise left to the reader, or a
        result asserted without proof.
      * start — if shown, the `position` of the content piece where the worked part BEGINS (often a
        piece that is or starts with "Proof", "Solution", "Derivation"). Everything before it is the
        statement or the question; that piece and everything after are the worked part. Use -1 when
        nothing is shown.
      * poses_task — does this block POSE a task for the reader to carry out (an exercise, a
        question, "find", "compute", "show that", "prove that")? True for an exercise or a worked
        example's question. False for a block that ASSERTS a claim as established (a theorem, a law).
      * procedure_type — one lowercase word for what the worked part is or would be, in the book's own
        vocabulary: "proof", "solution", "derivation", "calculation", "algorithm", … There is no fixed
        list.

    Judge only from the text given. Do not decide from what the block is called.
    """

    content: list[ContentPiece] = dspy.InputField(
        description="The block's content pieces, in order."
    )
    has_work: bool = dspy.OutputField(
        description='True if the block contains or calls for a worked derivation.'
    )
    shown: bool = dspy.OutputField(
        description='True if the worked steps are present in the text.'
    )
    start: int = dspy.OutputField(
        description='Content position where the worked part begins, or -1 if none shown.'
    )
    poses_task: bool = dspy.OutputField(
        description='True if the block poses a task for the reader rather than asserting a claim.'
    )
    procedure_type: str = dspy.OutputField(
        description='One lowercase word for the kind of worked part (proof / solution / derivation / …).'
    )


class Decompose(dspy.Signature):
    r"""
    Segment a WORKED DERIVATION — a proof, a solution, a calculation — into its ordered steps and
    label each with the role it plays. Return an ordered list of {description, action}.

    THE ROLES (choose exactly one per step):
      * premise — a setup step fixing objects or notation ("Let $x \in H$.", "Write $n = 2k$.").
      * assumption — a supposition made for the argument ("Assume for contradiction that ...").
      * lemma — a step that invokes or states an auxiliary result used in the argument.
      * corollary — a step that invokes a corollary or immediate consequence.
      * deduction — a logical inference from what came before ("Hence ...", "Therefore $x = y$.").
      * calculation — a computational or algebraic step (manipulating, evaluating, simplifying).
      * enumeration — a case split or itemized list of cases ("Case 1: ...", "Case 2: ...").
      * conclusion — the final step that establishes the claim or gives the answer ("This proves the
        theorem.", "as required", a QED / $\square$). Usually EXACTLY ONE, at the end.

    PARTITION (critical): the steps must exactly PARTITION the content — every part belongs to
    EXACTLY ONE step, no repeats and no omissions; reading the descriptions in order, with nothing
    added or removed, must reproduce the content.

    DESCRIPTIONS: copy each step's text VERBATIM — reproduce all mathematics and LaTeX exactly as
    given, changing nothing.

    Domain-neutral: a physics derivation or a biology mechanism walkthrough is segmented the same way.
    """

    contents: str = dspy.InputField(
        description="The worked part's content (text + LaTeX)."
    )
    actions: list[str] = dspy.InputField(
        description='The allowed role labels; choose one per step.'
    )
    steps: list[models.BodySegment] = dspy.OutputField(
        description='Ordered {description, action} steps; descriptions concatenate back to the content.'
    )


class Create(dspy.Signature):
    r"""
    Work out a posed textbook task whose solution the book does not show, and return BOTH the worked
    solution and its ordered steps.

    Solve it the way the book's own worked examples do: at the level of the surrounding material,
    showing the reasoning rather than only the answer, in the same notation the task uses. Label each
    step with the role it plays, from the given list.

    If the task cannot be worked out from what is given — it depends on a figure, a dataset, or a
    preceding part that is not present — return an empty solution and no steps rather than guessing.
    An honest gap is more useful than a fabricated derivation.
    """

    task: str = dspy.InputField(
        description='The posed task (text + LaTeX), with its shared instruction if it has one.'
    )
    actions: list[str] = dspy.InputField(
        description='The allowed role labels; choose one per step.'
    )
    solution: str = dspy.OutputField(
        description='The worked solution, or an empty string if it cannot be worked out from what is given.'
    )
    steps: list[models.BodySegment] = dspy.OutputField(
        description='Ordered {description, action} steps of the solution. Empty if there is no solution.'
    )


class Assessment(BaseModel):
    """The assessment pass's routing decision for one entity."""

    has_work: bool = False
    shown: bool = False
    start: int = -1
    poses_task: bool = False
    procedure_type: str = 'procedure'


class Module(dspy.Module):
    """Runs the assessment, decomposition, and creation passes."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.assess = dspy.Predict(Assess)
        self.decompose = dspy.ChainOfThought(Decompose)
        self.create = dspy.ChainOfThought(Create)
        self.set_lm(language_model or llm.text_lm())

    async def assessment(self, contents: list[str]) -> Assessment:
        """Returns the routing decision for one entity: is there work, is it shown, and where."""
        result = await self.assess.acall(
            content=[
                ContentPiece(position=k, text=text)
                for k, text in enumerate(contents)
            ]
        )
        return Assessment(
            has_work=bool(result.has_work),
            shown=bool(result.shown),
            start=(result.start if isinstance(result.start, int) else -1),
            poses_task=bool(result.poses_task),
            procedure_type=(
                _normalize_type(result.procedure_type) or 'procedure'
            ),
        )

    async def steps(self, contents: str) -> list[models.BodySegment]:
        """Segments a worked derivation into role-labelled steps."""
        result = await self.decompose.acall(
            contents=contents, actions=STEP_ACTIONS
        )
        return [
            step for step in (result.steps or []) if step.action in STEP_ACTIONS
        ]

    async def solve(self, task: str) -> tuple[str, list[models.BodySegment]]:
        """Works out a posed task the book leaves unsolved, returning its solution and steps."""
        result = await self.create.acall(task=task, actions=STEP_ACTIONS)
        solution = (result.solution or '').strip()
        if not solution:
            return '', []
        steps = [
            step for step in (result.steps or []) if step.action in STEP_ACTIONS
        ]
        return solution, steps


def _normalize_type(value: str | None) -> str | None:
    """The induced procedure type as a lowercase token, or None when nothing usable came back — the
    same normalization the entity type gets, and for the same reason (it keys a property index)."""
    name = ' '.join((value or '').split()).lower()
    return name or None


def task_text(entity: models.Entity) -> str:
    """The text a created procedure is worked out from: the entity's statement, prefixed by the
    shared instruction of its exercise group when it has one. The instruction is what carries the
    ask for a bare "12. $x^2 + 3x + 2$" exercise, so completion needs it even though extraction of a
    *shown* solution never did."""
    parts = ([entity.instruction] if entity.instruction else []) + list(
        entity.contents
    )
    return '\n\n'.join(part for part in parts if part and part.strip())


async def find_procedure(
    entity: models.Entity, module: Module | None = None
) -> models.Entity:
    """Extract, create, defer, or skip the procedure of one entity, in place.

    On **extract**, ``entity.contents`` is trimmed to the statement half and the worked half becomes
    a ``models.Procedure``; both halves are always kept, so a wrong boundary shifts a piece but never
    loses content. On **create**, ``contents`` is untouched and the generated procedure is marked
    ``generated``. On **defer** and **skip**, nothing changes. Returns the same entity.
    """
    module = module or Module()
    if not entity.contents:
        return entity
    assessment = await module.assessment(entity.contents)
    if not assessment.has_work:
        return entity  # skip: nothing to work out

    if assessment.shown:
        boundary = assessment.start
        if not 0 < boundary < len(entity.contents):
            return entity  # claimed shown but gave no usable boundary — leave it whole
        statement = entity.contents[:boundary]
        worked = entity.contents[boundary:]
        blob = '\n\n'.join(piece for piece in worked if piece.strip())
        steps = await module.steps(blob) if blob else []
        entity.contents = statement
        entity.procedures = [
            models.Procedure(
                type=assessment.procedure_type, contents=worked, steps=steps
            )
        ]
        return entity

    if assessment.poses_task:
        solution, steps = await module.solve(task_text(entity))
        if solution:
            entity.procedures = [
                models.Procedure(
                    type=assessment.procedure_type,
                    contents=[solution],
                    steps=steps,
                    generated=True,
                )
            ]
        return entity

    return entity  # defer: an asserted claim whose proof is absent


class ProcedureFinderNode:
    """Extracts (or creates) each block entity's procedure, in place.

    Runs after the universal attributor — it reads ``contents``, which the attributor fills — and
    after the instruction distributor, whose shared directive a created solution needs to know what
    is being asked. The per-entity passes are independent, so they run concurrently; the enriched
    entities are written back to the ``block_entities`` channel."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Extracts or creates each block entity's procedure, in place."""
        entities = state.get('block_entities', [])
        if entities:
            await asyncio.gather(
                *(find_procedure(entity, self.module) for entity in entities)
            )
        return {'block_entities': entities}
